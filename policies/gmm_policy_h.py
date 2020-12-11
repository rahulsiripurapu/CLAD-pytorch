from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from utils.mlp import MLP

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20
LOG_W_CAP_MIN = -10
EPS = 1e-6
#TODO: move these to config


class GMMPolicy(MLP):
    """Gaussian Mixture Model policy"""

    def __init__(self, env_spec, K=2, hidden_layer_sizes=[100, 100], reg=0.001,
                 squash=True, qf=None, device='cpu', reparametrization=False): #TODO use squash
        self._hidden_layers = hidden_layer_sizes
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._K = K
        self._is_deterministic = False
        super().__init__([self._Ds, *hidden_layer_sizes, K * (2 * self._Da + 1)],
                         output_activation=None)
        self._qf = qf
        self._reg = reg
        self._device = device
        self._normal = D.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self._reparametrization = reparametrization

    def unpack_dist_params(self,outputs):
        w_and_mu_and_logsig_t = torch.reshape(
            outputs, shape=(-1, self._K, 2 * self._Da + 1))  # Batchsize*K*(2*Da+1) for means and variances of action
        # distribution(diagonal) plus mixture weights for each mixture

        log_w_t = w_and_mu_and_logsig_t[..., 0]  # Batchsize*K
        mu_t = w_and_mu_and_logsig_t[..., 1:1 + self._Da]  # Batchsize*K*Da
        log_sig_t = w_and_mu_and_logsig_t[..., 1 + self._Da:]  # Batchsize*K*Da

        log_sig_t = torch.clamp(log_sig_t, min=LOG_SIG_CAP_MIN, max=LOG_SIG_CAP_MAX)

        log_w_t = torch.clamp(log_w_t, min=LOG_W_CAP_MIN)

        return log_w_t,mu_t,log_sig_t

    def create_gmm(self,log_w_t,mu_t,log_sig_t):
        w_t = torch.softmax(log_w_t,dim=-1)
        z = torch.multinomial(w_t,1) #Batchsizex1
        mu = mu_t[torch.arange(z.size(0)),z.squeeze()] #Batchsize x Da
        sig = torch.exp(log_sig_t[torch.arange(z.shape[0]),z.squeeze()]) #Batchsize x Da
        random = self._normal.sample(mu.shape).squeeze(-1).detach()
        if not self._device == 'cpu':
            random = random.cuda(self._device)
        sample = mu + sig * random # Batchsize x Da
        if not self._reparametrization:
            sample = sample.detach()
        prob_z = (sample[:,None,:]-mu_t)*torch.exp(-log_sig_t) # Batchsize x K x Da
        quad = -0.5*torch.sum(prob_z**2,dim=-1) #Batchsize x K
        full_var = torch.sum(log_sig_t,dim=-1) #Batchsize x K
        log_z = full_var + 0.5*mu_t.shape[-1]*np.log(2*np.pi)
        log_prob = quad - log_z
        log_prob_t = torch.logsumexp(log_prob + log_w_t,dim=1)-torch.logsumexp(log_w_t,dim=1)#Batchsize
        return sample,log_prob_t

    def forward(self, inputs):
        # Must return Mixturee distributions, actions, log_probs and reg_loss
        # TODO verify inputs dimensions
        # TODO ensure that inputs is on same gpu
        # assert inputs.dtype == torch.float32
        outputs = super().forward(inputs)
        log_w_t,mu_t,log_sig_t = self.unpack_dist_params(outputs)
        actions,log_probs = self.create_gmm(log_w_t,mu_t,log_sig_t)
        action_samples = torch.tanh(actions)
        corr = (2 * (np.log(2) - actions - torch.nn.functional.softplus(-2 * actions))).sum(axis=1)
        #corr = torch.sum(torch.log(1 - action_samples ** 2 + EPS), dim=1)
        assert not torch.isnan(corr).any() and not torch.isinf(corr).any()
        # logprobs require pure actions without squash correction.
        # TODO assert actions shape is Batchsize*Da.
        reg_loss = 0
        reg_loss += self._reg * 0.5 * torch.mean(log_sig_t ** 2)
        reg_loss += self._reg * 0.5 * torch.mean(mu_t ** 2)
        return (log_w_t,mu_t,log_sig_t), action_samples, log_probs, corr, reg_loss

    def get_actions(self, inputs):
        # TODO: if deterministic should return mean, else tanh of action?
        # inputs must be 2 dimensional, batchsize x statesize # TODO: add assertion
        # must return only action samples
        # TODO: Env does not support actions so must be single action
        inputs = torch.FloatTensor(inputs)
        if not self._device == 'cpu':
            # Neeeded because inputs are not on GPU during sample collection
            # in sanity check TODO: Sanity check is not the place for collecting samples.
            inputs = inputs.cuda(self._device)

        outputs = super().forward(inputs)
        log_w_t, mu_t, log_sig_t = self.unpack_dist_params(outputs)
        if(self._is_deterministic): #TODO implement deterministic for batch inputs(fix argmax)
            action = torch.tanh(mu_t.reshape([-1,self._Da])) # KxDa (or Batchsize*K x Da)
            q_vals = self._qf(inputs.expand((action.shape[0],-1)),action)
            q_vals = q_vals.detach().cpu().numpy()
            action = action.detach().cpu().numpy()
            return action[np.argmax(q_vals)].squeeze()

        actions_u,log_probs = self.create_gmm(log_w_t, mu_t, log_sig_t)
        actions = torch.tanh(actions_u).detach()
        # assert actions.shape == (inputs.shape[0], self._Da)
        return actions.squeeze().cpu().numpy()

    @staticmethod
    def _squash_correction(t):
        """receives action samples from gmm of shape batchsize x dim_action. For each action, the log probability
         correction requires a product by the inverse of the jacobian determinant. In log, it reduces to a sum, including
         the determinant of the diagonal jacobian. Adding epsilon to avoid overflow due to log
         Should return a tensor of batchsize x 1"""
        # TODO: Refer to OpenAI implementation for more numerically stable correction
        # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
        return torch.sum(torch.log(1 - t ** 2 + EPS), dim=1)

    @contextmanager
    def deterministic(self, set_deterministic=True):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
            to during the context. The value will be reset back to the previous
            value when the context exits.
        """
        current = self._is_deterministic
        self._is_deterministic = set_deterministic
        yield
        self._is_deterministic = current

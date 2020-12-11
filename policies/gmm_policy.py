from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from utils.mlp import MLP

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -5  # OpenAI uses -20.
LOG_W_CAP_MIN = -10
EPS = 1e-6


# TODO: move these to config


class GMMPolicy(MLP):
    """Gaussian Mixture Model policy"""

    def __init__(self, env_spec, K=2, hidden_layer_sizes=[100, 100], reg=0.001,
                 squash=True, qf=None, device='cpu'):  # TODO use squash
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

    def unpack_dist_params(self, outputs):
        w_and_mu_and_logsig_t = torch.reshape(
            outputs, shape=(-1, self._K, 2 * self._Da + 1))  # Batchsize*K*(2*Da+1) for means and variances of action
        # distribution(diagonal) plus mixture weights for each mixture

        log_w_t = w_and_mu_and_logsig_t[..., 0]  # Batchsize*K
        mu_t = w_and_mu_and_logsig_t[..., 1:1 + self._Da]  # Batchsize*K*Da
        log_sig_t = w_and_mu_and_logsig_t[..., 1 + self._Da:]  # Batchsize*K*Da

        log_sig_t = torch.clamp(log_sig_t, min=LOG_SIG_CAP_MIN, max=LOG_SIG_CAP_MAX)
        log_w_t = torch.clamp(log_w_t, min=LOG_W_CAP_MIN)

        return log_w_t, mu_t, log_sig_t

    def create_gmm(self, log_w_t, mu_t, log_sig_t):
        mix = D.Categorical(logits=log_w_t)  # Batchsize x K
        # refer https://github.com/pytorch/pytorch/pull/22742/files
        comp = D.Independent(D.Normal(mu_t, torch.exp(log_sig_t)), 1)
        # Individual Distribution = Batchsize x K x Da
        return D.MixtureSameFamily(mix, comp)

    def forward(self, inputs):
        # Must return Mixturee distributions, actions, log_probs and reg_loss
        # TODO verify inputs dimensions
        # TODO ensure that inputs is on same gpu
        # assert inputs.dtype == torch.float32
        outputs = super().forward(inputs)
        log_w_t, mu_t, log_sig_t = self.unpack_dist_params(outputs)
        gmm = self.create_gmm(log_w_t, mu_t, log_sig_t)
        actions = gmm.sample()  # Batchsize x Da
        # logprobs require pure actions without squash correction.
        action_samples = torch.tanh(actions)
        # Probability of squashed action is not same as probability of unsquashed action.
        corr = self._squash_correction(action_samples)
        assert not torch.isnan(corr).any() and not torch.isinf(corr).any()
        # correction must be subtracted from log_probs as we need inverse of jacobian determinant.
        # TODO assert actions shape is Batchsize*Da.
        reg_loss = 0
        reg_loss += self._reg * 0.5 * torch.mean(log_sig_t ** 2)
        reg_loss += self._reg * 0.5 * torch.mean(mu_t ** 2)
        return gmm, action_samples, gmm.log_prob(actions), corr, reg_loss

    def get_actions(self, inputs):
        # TODO: if deterministic should return mean, else tanh of action?
        # inputs must be 2 dimensional, batchsize x statesize # TODO: add assertion
        # must return only action samples
        # TODO: Env does not support actions so must be single action
        inputs = torch.FloatTensor(inputs)
        batch_size = inputs.shape[0]
        if not self._device == 'cpu':
            # Neeeded because inputs are not on GPU during sample collection
            # in sanity check TODO: Sanity check is not the place for collecting samples.
            inputs = inputs.cuda(self._device)
        outputs = super().forward(inputs)
        log_w_t, mu_t, log_sig_t = self.unpack_dist_params(outputs)
        if self._is_deterministic:
            action = torch.tanh(mu_t.reshape([-1, self._K, self._Da]))  # KxDa (or Batchsize*K x Da)
            # TODO expanding inputs causes waste of input compute in layer
            q_vals = self._qf(inputs.unsqueeze(1).expand((-1, self._K, -1)).reshape([-1, self._Ds]),
                              action.reshape([-1, self._Da]))
            q_vals = q_vals.detach().cpu().reshape([-1, self._K]).numpy() # Batchsize * K
            action = action.detach().cpu().reshape([-1, self._K, self._Da]).numpy() # Batchsize * K * Da
            # ,np.arange(self._Da).repeat(batch_size).reshape(self._Da,batch_size).transpose()
            return action[np.arange(batch_size), np.argmax(q_vals, axis=1)].squeeze() # Batchsize * Da

        gmm = self.create_gmm(log_w_t, mu_t, log_sig_t)
        actions = torch.tanh(gmm.sample()).detach()
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

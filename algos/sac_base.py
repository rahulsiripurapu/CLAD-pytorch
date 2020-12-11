from typing import Tuple, List
import numpy as np
import dateutil.tz
import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

import spaces
from envs.env_selector import env_selector
from envs.option_wrapper import optionwrap
from envs.normalized_env import normalize
from policies.gmm_policy import GMMPolicy
from replay_buffers import SimpleReplayBuffer
from utils.config import Config
from utils.sampler import Sampler
from value_functions.value_function import ValueFunction

EPS = 1E-6


class SAC():

    def __init__(self, config: Config) -> None:
        self.hparams = config
        self.env = env_selector(self.hparams)  # TODO: ensure normalization is not required
        self.eval_env = env_selector(self.hparams, config.seed + 1)  # TODO: add functionality to optionwrap for DIAYN
        # TODO: check all config.names to ensure they are in dict
        self.Da = self.env.action_space.flat_dim
        self.Do = self.env.observation_space.flat_dim
        self.qf = ValueFunction(self.Do + self.Da, [config.layer_size, config.layer_size])
        # Constructs a value function mlp with Relu hidden non-linearities, no output non-linearity and with xavier
        # init for weights and zero init for biases.
        self.vf = ValueFunction(self.Do, [config.layer_size, config.layer_size])
        self.vf_target = ValueFunction(self.Do, [config.layer_size, config.layer_size])
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.pool = SimpleReplayBuffer(
            env_spec=self.env.spec,
            max_replay_buffer_size=config.max_pool_size,
        )  # create a replay buffer for state+skill and action.

        self.policy = GMMPolicy(
            env_spec=self.env.spec,
            K=config.K,
            hidden_layer_sizes=[config.layer_size, config.layer_size],
            qf=self.qf,
            reg=config.reg,
            device="cpu"
        )  # GMM policy with K mixtures, no reparametrization trick, regularization

        # self.policy.cuda(config.device)
        # self.vf.cuda(config.device)
        # self.qf.cuda(config.device)
        # self.vf_target.cuda(config.device)

        # TODO: add assertion to test qf of policy and qf of model.

        self.sampler = Sampler(self.env, config.max_path_length)

        self._policy_lr = config.lr
        self._qf_lr = config.lr
        self._vf_lr = config.lr
        # TODO fix varialbe naming with _
        self._scale_reward = config.scale_reward
        self._discount = config.discount
        self._tau = config.tau
        self.max_path_return = -np.inf
        self.last_path_return = 0
        self.val_path_return = 0
        self._scale_entropy = config.scale_entropy

        self._save_full_state = config.save_full_state
        # self.z = self.get_best_skill(self.policy, self.env, self.config.num_skills, self.config.max_path_length)
        # self.env.reset(None,self.z)

        # Runs on CPU as Models are transferred to GPU only by trainer which happens after the lightning model init.
        # Also the reason why wandb logger is not available
        self.pool.add_samples(self.sampler.sample(config.min_pool_size, self.policy))
        # self.optimizers = []
        # TODO: combining vf and policy, figure out more elegant way to have unlinked learning rates than as
        # a multiplication factor in the loss sum. Also figure out why having them separate doesn't increase
        # compute time by the expected
        self.optimizer_policy = optim.Adam(list(self.policy.parameters())  # +list(self.vf.parameters())
                                           , lr=self._policy_lr)
        self.optimizer_vf = optim.Adam(self.vf.parameters(), lr=self._vf_lr)
        self.optimizer_qf = optim.Adam(self.qf.parameters(), lr=self._qf_lr)
        self.optimizer = optim.Adam(list(self.policy.parameters())+
                                    list(self.vf.parameters())+
                                    list(self.qf.parameters()), lr=self._policy_lr)
        # torch.autograd.set_detect_anomaly(True)

    @staticmethod
    def _squash_correction(t):
        """receives action samples from gmm of shape batchsize x dim_action. For each action, the log probability
         correction requires a product by the inverse of the jacobian determinant. In log, it reduces to a sum, including
         the determinant of the diagonal jacobian. Adding epsilon to avoid overflow due to log
         Should return a tensor of batchsize x 1"""
        # TODO: Refer to OpenAI implementation for more numerically stable correction
        # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
        return torch.sum(torch.log(1 - (t ** 2) + EPS), dim=1)

    def train(self):
        for epoch in range(self.hparams.max_epochs):
            for step in range(self.hparams.epoch_length):

                samples = self.sampler.sample(1, self.policy)  # TODO remove magic numbers
                self.pool.add_samples(samples)
                # print(samples[0]['done'])
                if samples[0]['done'] or samples[0]['path_length'] == self.hparams.max_path_length:
                    self.max_path_return = max(self.max_path_return, samples[0]['path_return'])
                    self.last_path_return = samples[0]['path_return']

                batch = self.pool.random_batch(self.hparams.batch_size)
                states, rewards, actions, dones, next_states = torch.FloatTensor(
                    batch['observations']), torch.FloatTensor(batch['rewards']), torch.FloatTensor(
                    batch['actions']), torch.FloatTensor(batch['dones']), torch.FloatTensor(batch['next_observations'])
                # self.optimizer_policy.zero_grad()
                self.optimizer.zero_grad()
                distributions, action_samples, log_probs, reg_loss = self.policy(states)
                # print(log_probs.shape)
                # assert log_probs.shape == torch.Size([action_samples.shape[0]])
                # TODO: figure out why squash correction is not done in policy as kl_surrogate seems
                # to need uncorrected log probs?
                self.values = self.vf(states)
                # print(action_samples.shape,log_probs.shape,reg_loss.shape,states.shape) #TODO assert shapes

                with torch.no_grad():

                    self.log_targets = self.qf(states, action_samples)
                    # Probability of squashed action is not same as probability of unsquashed action.
                    corr = self._squash_correction(action_samples)
                    # print(log_probs.shape,corr.shape)
                    # assert not torch.isnan(corr).any() and not torch.isinf(corr).any()
                    # correction must be subtracted from log_probs as we need inverse of jacobian determinant.
                    self.scaled_log_pi = self._scale_entropy * (log_probs - corr)


                # self._vf_loss = 0.5 * torch.mean(
                #             (self.values - self.log_targets - self.scaled_log_pi) ** 2)
                ## How is this kl surrogate loss derived?
                self._kl_surrogate_loss = torch.mean(log_probs * (
                        self.scaled_log_pi - self.log_targets + self.values.detach()))
                self._policy_loss = reg_loss + self._kl_surrogate_loss

                # self._policy_loss.backward()
                # self.optimizer_policy.step()
                #
                # self.optimizer_vf.zero_grad()
                # self.values = self.vf(states)
                self._vf_loss = 0.5 * torch.mean(
                    (self.values - self.log_targets + self.scaled_log_pi) ** 2)



                # self._vf_loss.backward()
                # self.optimizer_vf.step()
                #
                # self.optimizer_qf.zero_grad()
                self.q_values = self.qf(states, actions)
                # assert (self.policy._qf(states,actions)==self.q_values).all()
                with torch.no_grad():
                    vf_next_target = self.vf_target(next_states)  # N
                    # self._vf_target_params = self._vf.get_params_internal()

                    ys = self._scale_reward * rewards + (1 - dones) * self._discount * vf_next_target  # N


                self._td_loss = 0.5 * torch.mean((ys - self.q_values) ** 2)

                #TODO COde not working, need to fix bug
                self.loss = self._policy_loss + self._vf_loss + self._td_loss
                self.loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    for vf, vf_targ in zip(self.vf.parameters(), self.vf_target.parameters()):
                        vf_targ.data.mul_(1 - self.hparams.tau)
                        vf_targ.data.add_((self.hparams.tau) * vf.data)


            print('train_loss: ', self._policy_loss.detach().numpy(),
                  'epoch: ', epoch,
                  # 'vf_loss': self._vf_loss,
                  # 'steps': torch.tensor(self.global_step),#.to(device),#Where did this global_step comee from is it PL inbuilt?
                  'max_return: ', (self.max_path_return),
                  'last_return: ', (self.last_path_return),
                  # 'gmm_means: ', torch.mean(distributions.component_distribution.mean).detach().numpy(),
                  # 'gmm_sigmas: ', torch.mean(distributions.component_distribution.stddev).detach().numpy(),
                  'vf_loss: ', self._vf_loss.detach().numpy(),
                  'vf_value: ', torch.mean(self.values).detach().numpy(),
                  'qf_loss: ', self._td_loss.detach().numpy(),
                  'rewards: ', torch.mean(rewards).detach().numpy(),
                  'actions: ', torch.mean(actions).detach().numpy(),
                  'qf_value: ', torch.mean(self.q_values).detach().numpy()
                  )

            state = self.eval_env.reset()
            # print("Running Validation")
            path_return = 0
            path_length = 0
            self.ims = []
            print(datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y-%m-%d-%H-%M-%S-%f-%Z'))
            # with self.policy.deterministic(True):
            #     for i in range(self.hparams.max_path_length):
            #         action = self.policy.get_actions(state.reshape((1, -1)))
            #         next_ob, reward, done, info = self.eval_env.step(action)
            #         if self.hparams.render_validation:
            #             self.ims.append(self.eval_env.render(mode='rgb_array'))
            #             # print(self.ims[0].shape)#config={'height':500,'width':500,'xpos':0,'ypos':0,'title':'validation'}
            #         # print(reward)
            #         state = next_ob
            #         path_return += reward
            #         path_length += 1
            #         if (done):
            #             break

            self.val_path_return = path_return
            print('path_return: ', path_return,
                  'path_length: ', path_length)

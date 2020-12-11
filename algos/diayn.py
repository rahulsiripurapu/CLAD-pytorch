from collections import OrderedDict
from typing import Tuple, List, Dict, Any
import numpy as np
import dateutil.tz
import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer

from algos.sac import SAC
from reward_functions.discriminator import Discriminator
from utils.config import Config
from utils.sampler import Sampler


class DIAYN(SAC):

    def __init__(self, config: Config) -> None:

        super().__init__(config)
        self.z = 0
        self._num_skills = self.hparams.num_skills
        # Env is already set to skill 0 upon init from env_selector but better to do to avoid rare cases
        self.env.reset(state=None, skill=self.z)
        self.eval_env.reset(state=None, skill=self.z)
        self.discriminator = Discriminator(self.Do - self._num_skills, [config.layer_size, config.layer_size],
                                           self._num_skills)
        self.modules.append(["Discriminator", self.discriminator])
        self._discriminator_lr = config.lr
        self._p_z = torch.FloatTensor(np.full(self._num_skills, 1.0 / self._num_skills))
        self.sampler.reset()

    def on_sanity_check_start(self):
        self.pool.add_samples(self.sampler.sample(self.hparams.min_pool_size, self.policy))
        print("Initialized Replay Buffer with %d samples" % self.pool.size)
        if self.on_gpu:
            self._p_z = self._p_z.cuda(self.hparams.device)
            print("Moving p_z to GPU")

    def on_epoch_start(self):
        self.z = self._sample_z()
        print("\nUsing skill :", self.z)
        self.env.reset(state=None, skill=self.z)
        self.sampler.reset()
        self.eval_env.reset(state=None, skill=self.z)

    def _split_obs(self, t):
        # TODO: verify that dim is 1, assert shape
        return torch.split(t, [self.Do - self._num_skills, self._num_skills], -1)

    def _sample_z(self):
        """Samples z from p(z), using probabilities in self._p_z."""
        return np.random.choice(self._num_skills, p=self._p_z.detach().cpu().numpy())

    def training_step(self, batch, batch_idx, optimizer_idx) -> OrderedDict:

        states, actions, _, dones, next_states = batch

        # print(self.pool.size,optimizer_idx,batch_idx,states[0])
        # print("Running train",states.shape,batch_idx,optimizer_idx)

        # TODO: vars are already floatTensors.
        # Train Policy
        if optimizer_idx == 0:
            samples = self.sampler.sample(1, self.policy)  # TODO remove magic numbers
            self.pool.add_samples(samples)

            if samples[0]['done'] or samples[0]['path_length'] == self.hparams.max_path_length:
                self.max_path_return = max(self.max_path_return, samples[0]['path_return'])
                self.last_path_return = samples[0]['path_return']

            distributions, action_samples, log_probs, corr, reg_loss = self.policy(states)
            assert log_probs.shape == torch.Size([action_samples.shape[0]])
            # TODO: figure out why squash correction is not done in policy as kl_surrogate seems
            # to need uncorrected log probs?
            self.values = self.vf(states)
            # print(action_samples.shape,log_probs.shape,reg_loss.shape,states.shape) #TODO assert shapes

            with torch.no_grad():
                self.log_targets = self.qf(states, action_samples)
                self.scaled_log_pi = self._scale_entropy * (log_probs - corr)

            ## How is this kl surrogate loss derived?
            self._kl_surrogate_loss = torch.mean(log_probs * (
                    self.scaled_log_pi - self.log_targets + self.values.detach()))
            self._policy_loss = reg_loss + self._kl_surrogate_loss
            self._vf_loss = 0.5 * torch.mean(
                (self.values - self.log_targets + self.scaled_log_pi) ** 2)
            # return OrderedDict({'loss': self._vf_loss,
            #                     'log': {'vf_loss': self._vf_loss,
            #                             'vf_value': torch.mean(self.values)},
            #                     'progress_bar': {'vf_loss': self._vf_loss,
            #                                      'vf_value': torch.mean(self.values)}})

            log = {'max_path_return': torch.tensor(self.max_path_return),
                   'train_loss': self._policy_loss,
                   'kl_loss': self._kl_surrogate_loss,
                   'reg_loss': reg_loss,
                   'gmm_means': torch.mean(distributions.component_distribution.mean),
                   'gmm_sigmas': torch.mean(distributions.component_distribution.stddev),
                   'vf_loss': self._vf_loss,
                   'vf_value': torch.mean(self.values),
                   'scaled_log_pi': torch.mean(self.scaled_log_pi).detach().cpu().numpy()
                   }
            status = {
                'train_loss': self._policy_loss,
                # 'vf_loss': self._vf_loss,
                # 'steps': torch.tensor(self.global_step),#.to(device),#Where did this global_step comee from is it PL inbuilt?
                'max_ret': torch.tensor(self.max_path_return),
                'last_ret': torch.tensor(self.last_path_return),
                'gmm_mu': torch.mean(distributions.component_distribution.mean),
                'gmm_sig': torch.mean(distributions.component_distribution.stddev),
                'vf_loss': self._vf_loss,
                'vf_mu': torch.mean(self.values)
            }

            return OrderedDict({'loss': self._policy_loss + self._vf_loss,
                                'log': log, 'progress_bar': status})

        # Train VF
        # if optimizer_idx == 1:
        # for param in self.vf.parameters():
        #     print(param.names, param.size(), param.requires_grad)
        # print("Donevf")
        # TODO: figure out way to avoid calling value function twice and just reuse previous call.
        # self.values = self.vf(states)

        # Train QF
        if optimizer_idx == 1:
            # for param in self.qf.parameters():
            #     print(param.names, param.size(), param.requires_grad)
            # print("Doneqf")
            self.q_values = self.qf(states, actions)

            # assert (self.policy._qf(states,actions)==self.q_values).all()
            with torch.no_grad():
                (obs, z) = self._split_obs(states)
                logits = self.discriminator(obs)  # N x num_skills
                skill = torch.argmax(z, dim=1)  # N
                reward = -1 * nn.CrossEntropyLoss(reduction='none')(logits, skill)  # N
                assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                p_z = torch.sum(self._p_z * z, dim=1)  # N
                log_p_z = torch.log(p_z + self.hparams.eps)
                if self.hparams.add_p_z:
                    reward -= log_p_z
                    assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                vf_next_target = self.vf_target(next_states)  # N

                ys = self._scale_reward * reward + (1 - dones) * self._discount * vf_next_target  # N

            self._td_loss = 0.5 * torch.mean((ys - self.q_values) ** 2)

            return OrderedDict(
                {'loss': self._td_loss,
                 'log': {'qf_loss': self._td_loss,
                         'qf_value': torch.mean(self.q_values),
                         'rewards': torch.mean(reward)},
                 'progress_bar': {'qf_loss': self._td_loss,
                                  'rewards': torch.mean(reward),
                                  'actions': torch.mean(actions),
                                  'qf_value': torch.mean(self.q_values)}})

        if optimizer_idx == 2:
            (obs, z) = self._split_obs(states)
            logits = self.discriminator(obs)
            skill = torch.argmax(z, dim=1)
            self._discriminator_loss = nn.CrossEntropyLoss()(logits, skill)

            return OrderedDict(
                {'loss': self._discriminator_loss,
                 'log': {'discriminator_loss': self._discriminator_loss,
                         },
                 'progress_bar': {'discriminator_loss': self._discriminator_loss,
                                  }})

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.hparams.save_pool and self.current_epoch == self.hparams.max_epochs - 1:
            checkpoint['pool'] = self.pool

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizers"""
        optimizers = []
        # TODO: combining vf and policy, figure out more elegant way to have unlinked learning rates than as
        # a multiplication factor in the loss sum. Also figure out why having them separate doesn't increase
        # compute time by the expected
        optimizers.append(optim.Adam(list(self.policy.parameters()) + list(self.vf.parameters())
                                     , lr=self._policy_lr))
        optimizers.append(optim.Adam(self.qf.parameters(), lr=self._qf_lr))
        optimizers.append(optim.Adam(self.discriminator.parameters(), lr=self._discriminator_lr))
        return optimizers

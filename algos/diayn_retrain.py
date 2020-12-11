import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import gc

import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer
import numpy as np
import dateutil.tz
import datetime

from algos.diayn import DIAYN
from policies.gmm_policy import GMMPolicy
from reward_functions.discriminator import Discriminator
from utils.config import Config
from value_functions.value_function import ValueFunction


class DIAYN_retrain(DIAYN):
    # Runs DIAYN with pretrained discriminator. Doesn't train the discriminator
    # Only discriminator must be initialized from checkpoint
    # Supports training on curriiculum, need to specify switching epochs in config
    # Distillers must also be initialized from checkpoint

    def __init__(self, config: Config) -> None:

        super().__init__(config)
        self.skilldata_train = [[] for i in range(self._num_skills)]
        # variable to store rewards for each skill after validation.
        self.skilldata_val = [[] for i in range(self._num_skills)]
        self.distiller = [
            Discriminator(self.Do - self._num_skills, [self.hparams.disc_size[i], self.hparams.disc_size[i]],
                          self._num_skills) for i in range(len(self.hparams.disc_size))]

    def on_sanity_check_start(self):
        if self.hparams.distill:
            self.stage = 0
            self.distiller.append(self.discriminator)
            self.discriminator = self.distiller[self.stage]
        print(self.discriminator.state_dict)
        # Reinitializing to wipe out values loaded from checkpoint.
        qf = ValueFunction(self.Do + self.Da, [self.hparams.layer_size, self.hparams.layer_size])
        self.qf.load_state_dict(qf.state_dict())
        vf = ValueFunction(self.Do, [self.hparams.layer_size, self.hparams.layer_size])
        self.vf.load_state_dict(vf.state_dict())
        self.vf_target.load_state_dict(self.vf.state_dict())
        # TODO figure out why vf doesn't need load state dict, ie doesnt throw tensors does not reequire grad error.
        policy = GMMPolicy(
            env_spec=self.env.spec,
            K=self.hparams.K,
            hidden_layer_sizes=[self.hparams.layer_size, self.hparams.layer_size],
            qf=self.qf,
            reg=self.hparams.reg,
            device=self.hparams.device
        )  # GMM policy with K mixtures, no reparametrization trick, regularization
        self.policy.load_state_dict(policy.state_dict())
        # Verified by loading trained checkpoint that policy qf and self.qf are the exact same
        # TODO hack to load only discriminator, reemove also is bleow line needed?
        self.modules = ["Policy", self.policy, "QF", self.qf, "VF", self.vf, "VF_Target", self.vf_target,
                        "Discriminator", self.discriminator]
        self.pool.add_samples(self.sampler.sample(self.hparams.min_pool_size, self.policy))
        print("Initialized Replay Buffer with %d samples" % self.pool.size)
        if self.on_gpu:
            self._p_z = self._p_z.cuda(self.hparams.device)
            for i in range(len(self.hparams.disc_size)):
                self.distiller[i].cuda(self.hparams.device)
            print("Moving p_z and distillers to GPU")

    def on_epoch_start(self):
        self.z = self._sample_z()
        print("\nUsing skill :", self.z)
        self.env.reset(state=None, skill=self.z)
        self.sampler.reset()
        self.eval_env.reset(state=None, skill=self.z)
        while self.current_epoch == self.hparams.switch_epoch[self.stage]:
            self.stage += 1
            self.discriminator = self.distiller[self.stage]
            print("\nSwitching to next stage %d with discriminator: " % self.stage, self.discriminator)


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
            #Q-V gives the current policy's advantage.
            self._policy_loss = reg_loss + self._kl_surrogate_loss
            self._vf_loss = 0.5 * torch.mean(
                (self.values - self.log_targets + self.scaled_log_pi) ** 2)
            # Average value of Value function is Q + the entropy bonus
            # (as q function does not record entropy bonuses).

            log = {'max_path_return': self.max_path_return,
                   'train_loss': self._policy_loss.detach().cpu().numpy(),
                   'kl_loss': self._kl_surrogate_loss.detach().cpu().numpy(),
                   'reg_loss': reg_loss.detach().cpu().numpy(),
                   'gmm_means': torch.mean(distributions.component_distribution.mean).detach().cpu().numpy(),
                   'gmm_sigmas': torch.mean(distributions.component_distribution.stddev).detach().cpu().numpy(),
                   'vf_loss': self._vf_loss.detach().cpu().numpy(),
                   'vf_value': torch.mean(self.values).detach().cpu().numpy(),
                   'scaled_log_pi': torch.mean(self.scaled_log_pi).detach().cpu().numpy()
                   }
            status = {
                'train_loss': self._policy_loss,
                # 'vf_loss': self._vf_loss,
                # 'steps': torch.tensor(self.global_step),#.to(device),#Where did this global_step comee from is it PL inbuilt?
                'max_ret': self.max_path_return,
                'last_ret': self.last_path_return,
                'gmm_mu': torch.mean(distributions.component_distribution.mean).detach().cpu().numpy(),
                'gmm_sig': torch.mean(distributions.component_distribution.stddev).detach().cpu().numpy(),
                'vf_loss': self._vf_loss,
                'vf_mu': torch.mean(self.values).detach().cpu().numpy()
            }

            return OrderedDict({'loss': self._policy_loss + self._vf_loss,
                                'log': log, 'progress_bar': status})

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
                print(logits.shape,skill.shape,skill,logits,skill.type(),logits.type())
                assert 0==1
                reward = -1 * nn.CrossEntropyLoss(reduction='none')(logits, skill)  # N
                reward = torch.clamp(reward, min=-8)
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
                 'log': {'qf_loss': self._td_loss.detach().cpu().numpy(),
                         'qf_value': torch.mean(self.q_values).detach().cpu().numpy(),
                         'rewards': torch.mean(reward).detach().cpu().numpy()},
                 'progress_bar': {'qf_loss': self._td_loss,
                                  'rewards': torch.mean(reward).detach().cpu().numpy(),
                                  'actions': torch.mean(actions).detach().cpu().numpy(),
                                  'qf_value': torch.mean(self.q_values).detach().cpu().numpy()}})

    def validation_epoch_end(self, outputs) -> OrderedDict:
        gc.collect()
        state = self.eval_env.reset()
        print(datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y-%m-%d-%H-%M-%S-%f-%Z'))
        # print("Running Validation")
        path_return = 0
        path_length = 0
        self.ims = []
        rewards = []
        with self.policy.deterministic(self.hparams.deterministic_eval):
            for i in range(self.hparams.max_path_length):
                action = self.policy.get_actions(state.reshape((1, -1)))
                next_ob, reward, done, info = self.eval_env.step(action)
                (obs, z) = self._split_obs(torch.FloatTensor(next_ob)[None, :])
                if self.on_gpu:
                    # Neeeded because inputs are not on GPU during sample collection
                    # in sanity check TODO: Sanity check is not the place for collecting samples.
                    obs = obs.cuda(self.hparams.device)
                    z = z.cuda(self.hparams.device)
                logits = self.discriminator(obs)  # N x num_skills
                skill = torch.argmax(z, dim=-1)  # N
                reward = -1 * nn.CrossEntropyLoss(reduction='none')(logits, skill)  # N
                reward = torch.clamp(reward, min=-8)
                assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                p_z = torch.sum(self._p_z * z, dim=-1)  # N
                log_p_z = torch.log(p_z + self.hparams.eps)
                if self.hparams.add_p_z:
                    reward -= log_p_z
                    assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                # self.eval_env.render(mode='human')
                if self.hparams.render_validation:
                    self.ims.append(self.eval_env.render(mode='rgb_array'))
                    # print(self.ims[0].shape)#config={'height':500,'width':500,'xpos':0,'ypos':0,'title':'validation'}
                # print(reward)
                reward = reward.cpu()
                rewards.append(reward.numpy())
                state = next_ob
                path_return += reward
                path_length += 1
                if (done):
                    break

        mean_reward = np.mean(np.asarray(rewards))
        # self.reward_map[self.z,self.current_epoch:]=mean_reward
        rewards = [mean_reward]
        rewards.append(self.current_epoch)
        self.skilldata_val[self.z].append(np.asarray(rewards))
        self.val_path_return = path_return  # TODO : remove printcall back for this, already printed in progress bar
        return OrderedDict({'log': {'path_return': path_return,
                                    'path_length': path_length},
                            'progress_bar': {'val_ret': path_return,
                                             'path_len': path_length}})

    def on_train_end(self) -> None:
        avg_ret = np.zeros(len(self.distiller))
        for i in range(self._num_skills):
            print("Generating video for skill ", i)
            ims, rews, ret = self.run_skill(i)
            avg_ret += ret
        print("Average path return: ", avg_ret / self._num_skills)

    def run_skill(self,skill):
        ims = []
        rewards = [] #[[] for i in range(len(self.distiller))]
        path_return = np.zeros(len(self.distiller))
        for j in range(self.hparams.num_runs):
            state = self.env.reset(state=None, skill=skill)
            # print(datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y-%m-%d-%H-%M-%S-%f-%Z'))
            # print("Running Validation")

            path_length = 0


            with self.policy.deterministic(self.hparams.deterministic_eval):
                for k in range(self.hparams.max_path_length):
                    action = self.policy.get_actions(state.reshape((1, -1)))
                    next_ob, _, done, info = self.env.step(action)
                    (obs, z) = self._split_obs(torch.FloatTensor(next_ob)[None, :])
                    if self.on_gpu:
                        # Neeeded because inputs are not on GPU during sample collection
                        # in sanity check TODO: Sanity check is not the place for collecting samples.
                        obs = obs.cuda(self.hparams.device)
                        z = z.cuda(self.hparams.device)
                    logits = [self.distiller[i](obs) for i in range(len(self.distiller))]  # N x num_skills
                    skillz = torch.argmax(z, dim=-1)  # N
                    reward = torch.FloatTensor([-1 * nn.CrossEntropyLoss(reduction='none')(logits[i], skillz) for i in range(len(self.distiller))])  # N
                    reward = torch.clamp(reward, min=-8)
                    assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                    p_z = torch.sum(self._p_z * z, dim=-1)  # N
                    log_p_z = torch.log(p_z + self.hparams.eps)
                    if self.hparams.add_p_z:
                        reward -= log_p_z
                        assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                    # if self.hparams.render_validation:
                    #     ims.append(cv2.resize(self.env.render(mode='rgb_array'),(500,500)))
                    #     # print(self.ims[0].shape)#config={'height':500,'width':500,'xpos':0,'ypos':0,'title':'validation'}
                    #     # print(reward)
                    state = next_ob
                    path_return += reward.numpy()
                    path_length += 1
                    rewards.append(reward)
                    if (done):
                        break
            # print(path_return)


        return ims, rewards, path_return/self.hparams.num_runs




    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizers"""
        optimizers = []
        # TODO: combining vf and policy, figure out more elegant way to have unlinked learning rates than as
        # a multiplication factor in the loss sum. Also figure out why having them separate doesn't increase
        # compute time by the expected
        optimizers.append(optim.Adam(list(self.policy.parameters()) + list(self.vf.parameters())
                                     , lr=self._policy_lr))
        optimizers.append(optim.Adam(self.qf.parameters(), lr=self._qf_lr))
        return optimizers

# import collections
# import resource

import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List, Dict, Any

import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer
import numpy as np
import dateutil.tz
import datetime
# import gc

from algos.diayn import DIAYN
from policies.gmm_policy import GMMPolicy
from reward_functions.discriminator import Discriminator
from utils.config import Config
from value_functions.value_function import ValueFunction


class DIAYN_distill(DIAYN):
    # Distills the diayn discriminator into multiple discriminators of smaller size
    # Creates new distillers into which the discriminator is distilled
    # If full pool is given, distillation is done using given pool
    # Else fresh data is collected for all skills from the trained agent in the checkpoint.

    # def get_cpu_mem(self):
    #     return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #
    # def get_num_of_tensors(self):
    #     tensors_num = 0
    #     sizes = collections.Counter()
    #     for obj in gc.get_objects():
    #         try:
    #             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #                 tensors_num += 1
    #                 sizes[obj.size()] += 1
    #         except:
    #             pass
    #     res = sizes[torch.Size([])]
    #     return res


    def __init__(self, config: Config) -> None:

        super().__init__(config)
        self.skilldata_train = [[] for i in range(self._num_skills)]
        self.skilldata_val = [[] for i in range(self._num_skills)]
        #TODO verify naming in checkpoint dict
        self.distiller = [Discriminator(self.Do - self._num_skills, [self.hparams.disc_size[i], self.hparams.disc_size[i]],
                                      self._num_skills) for i in range(len(self.hparams.disc_size))]
        # self.reward_map = np.zeros((self._num_skills,self.hparams.max_epochs))-4

    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #
    # def on_pre_performance_check(self) -> None:
    #
    # def on_train_start(self) -> None:

    def on_sanity_check_start(self):
        # TODO hack to load only discriminator, reemove also is bleow line needed?
        self.modules = ["Policy", self.policy, "QF", self.qf, "VF", self.vf, "VF_Target", self.vf_target, "Discriminator", self.distiller]
        print("Existing pool data: ", self.pool.size)
        if self.pool.size < self.hparams.max_pool_size:
            for i in range(self._num_skills):
                self.z = i
                print("\nSampling skill :", self.z)
                self.env.reset(state=None, skill=self.z)
                self.sampler.reset()
                # self.eval_env.reset(state=None, skill=self.z)
                self.pool.add_samples(self.sampler.sample(self.hparams.min_pool_size, self.policy))
            print("Initialized Replay Buffer with %d samples" % self.pool.size)
        if self.on_gpu:
            self._p_z = self._p_z.cuda(self.hparams.device)
            for i in range(len(self.hparams.disc_size)):
                self.distiller[i].cuda(self.hparams.device)
            print("Moving p_z and distillers to GPU")

    def training_step(self, batch, batch_idx) -> OrderedDict:

        states, actions, _, dones, next_states = batch
        # print("\nEpoch ", self.current_epoch, " Number of tensors :", self.get_num_of_tensors(), " CPU mem usage ",
        #       self.get_cpu_mem())
        # print(self.pool.size,optimizer_idx,batch_idx,states[0])
        # print("Running train",states.shape,batch_idx,optimizer_idx)

        # TODO: vars are already floatTensors.
        # Train Discriminator

        if self.pool.size < self.hparams.max_pool_size:
            samples = self.sampler.sample(1, self.policy)  # TODO remove magic numbers
            self.pool.add_samples(samples)

            if samples[0]['done'] or samples[0]['path_length'] == self.hparams.max_path_length:
                self.max_path_return = max(self.max_path_return, samples[0]['path_return'])
                self.last_path_return = samples[0]['path_return']

        (obs, z) = self._split_obs(states)
        logits = [self.distiller[i](obs) for i in range(len(self.hparams.disc_size))]
        skill = torch.argmax(z, dim=1)
        distiller_losses = [nn.CrossEntropyLoss()(logits[i], skill) for i in range(len(self.hparams.disc_size))]
        self._distiller_loss = 0
        for i in range(len(self.hparams.disc_size)):
            self._distiller_loss += distiller_losses[i]
        for i in range(len(self.hparams.disc_size)):
            distiller_losses[i] = distiller_losses[i].cpu().detach().numpy()

        return OrderedDict(
            {'loss': self._distiller_loss,
             'log': {
                 'max_path_return': self.max_path_return,
                 'discriminator_loss': self._distiller_loss.detach().cpu().numpy(),
                     },
             'progress_bar': {
                 'max_ret': self.max_path_return,
                 'last_ret': self.last_path_return,
                 'discriminator_loss': distiller_losses,
                              }})

    def validation_epoch_end(self, outputs) -> OrderedDict:
        # Validates agents with distiller of smallest size
        # gc.collect()
        # print("\nEpoch ", self.current_epoch, " Number of tensors :", self.get_num_of_tensors(), " CPU mem usage ",self.get_cpu_mem())
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
                (obs, z) = self._split_obs(torch.FloatTensor(next_ob)[None,:])
                if self.on_gpu:
                    # Neeeded because inputs are not on GPU during sample collection
                    # in sanity check TODO: Sanity check is not the place for collecting samples.
                    obs = obs.cuda(self.hparams.device)
                    z = z.cuda(self.hparams.device)
                logits = self.distiller[-1](obs)  # N x num_skills
                skill = torch.argmax(z, dim=-1)  # N
                reward = -1 * nn.CrossEntropyLoss(reduction='none')(logits, skill)  # N
                reward = torch.clamp(reward,min=-8)
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
                reward = reward.cpu().detach()
                rewards.append(reward.numpy())
                state = next_ob
                path_return += reward
                path_length += 1
                if(done):
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

    def on_batch_end(self) -> None:
        pass

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for i in range(len(self.hparams.disc_size)):
            checkpoint['distiller%d'%i]=self.distiller[i].state_dict()
        print(checkpoint.keys())

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizers"""
        optimizers = []
        # list(self.policy.parameters())
        self.params = []
        for i in range(len(self.hparams.disc_size)):
            self.params += list(self.distiller[i].parameters())
        optimizers.append(optim.Adam(self.params, lr=self._discriminator_lr))
        return optimizers

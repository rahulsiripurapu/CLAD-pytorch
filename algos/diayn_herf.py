from collections import OrderedDict
from typing import Tuple, List

import cv2
import numpy as np
import dateutil.tz
import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer

from algos.sac import SAC
from envs.env_selector import env_selector
from reward_functions.herf_hc import HERF
from reward_functions.discriminator import Discriminator
from utils.config import Config
from utils.sampler import Sampler


class DHERF(SAC):

    def __init__(self, config: Config) -> None:

        super().__init__(config)
        self.z = 0
        self._num_skills = self.hparams.num_skills
        self.env.set_reward_fn(HERF())
        self.eval_env.set_reward_fn(HERF())
        self.env.reset(state=None, skill=self.z)
        self.eval_env.reset(state=None, skill=self.z)
        self.batch_return = 0
        self.single_skill = config.single_skill
        self.double_skill = config.double_skill
        self.skilldata_val = [[] for i in range(self._num_skills)]
        if self.single_skill is None:
            self.batch_env = [env_selector(self.hparams, config.seed + 1) for i in range(self._num_skills)]
            for i in range(self._num_skills):
                self.batch_env[i].set_reward_fn(HERF())
                self.batch_env[i].reset(skill=i)
        # TODO HERF only supports upto 25 skills, uses modulo beyond that.
        self.discriminator = Discriminator(self.Do - self._num_skills, [config.layer_size, config.layer_size],
                                           self._num_skills)
        self.distiller = [
            Discriminator(self.Do - self._num_skills, [self.hparams.disc_size[i], self.hparams.disc_size[i]],
                          self._num_skills) for i in range(len(self.hparams.disc_size))]

        self.sampler.reset()
        self._p_z = torch.FloatTensor(np.full(self._num_skills, 1.0 / self._num_skills))

    def _match_best_skills(self):
        # initialize table to store option, reward fn index, and return
        for param in self.policy.parameters():
            wp_ref = param
            break
        for param in self.vf.parameters():
            wv_ref = param
            break
        for param in self.qf.parameters():
            wq_ref = param
            break
        wp = wp_ref.clone()
        wv = wv_ref.clone()
        wq = wq_ref.clone()
        reward_table = np.zeros((self._num_skills * self._num_skills, 3))
        # fills 000....[50]1111...[50]......49
        fill = np.arange(self._num_skills).repeat(self._num_skills)
        reward_table[:, 0] = fill  # options
        # fills 012345...49012345....49....
        reward_table[:, 1] = fill.reshape((-1, self._num_skills)).transpose().flatten()  # reward fns

        states = np.zeros((self._num_skills, self.Do))
        next_ob = np.zeros((self._num_skills, self.Do))
        reward = np.zeros((self._num_skills, self._num_skills))  # Options x Reward fns
        # TODO does not support done.
        # done = np.zeros(self._num_skills)
        # info = [[] for i in range(self._num_skills)]
        # allocatedo = [False] * self._num_skills  # whether option i is allocated to reward
        allocatedr = [False] * self._num_skills  # whether reward i is allocated to option
        for k in range(2):
            for i in range(self._num_skills):
                states[i] = self.batch_env[i].reset()
                print("batch env %d, skill %d"%(i,self.batch_env[i].reward_fn.skill))
            with self.policy.deterministic(self.hparams.deterministic_eval):
                for j in range(self.hparams.max_path_length):
                    actions = self.policy.get_actions(states)
                    for i in range(self._num_skills):
                        next_ob[i], reward[i], _, _ = self.batch_env[i].m_step(actions[i])
                    states = next_ob
                    reward_table[:, 2] += reward.flatten()  # accumulate returns for each option reward combination

        # sort reward_table based on returns
        reward_table = reward_table[reward_table[:, 2].argsort()[::-1]]
        skill_table = np.zeros(self._num_skills)
        j = 0
        for i in range(self._num_skills * self._num_skills):
            if not allocatedr[int(reward_table[i, 1])]:
                skill_table[j] = j#reward_table[i, 1]
                wp[:,self.Do - self._num_skills + int(reward_table[i, 1])] = wp_ref[:,self.Do -self._num_skills + int(reward_table[i, 0])]
                wv[:,self.Do - self._num_skills + int(reward_table[i, 1])] = wv_ref[:,self.Do -self._num_skills + int(reward_table[i, 0])]
                wq[:,self.Do - self._num_skills + int(reward_table[i, 1])] = wq_ref[:,self.Do -self._num_skills + int(reward_table[i, 0])]
                print("Allocating skill %d with reward fn %d with retrun %f" % (
                reward_table[i, 0], reward_table[i, 1], reward_table[i, 2]))
                j += 1
                # allocatedo[int(reward_table[i, 0])] = True
                allocatedr[int(reward_table[i, 1])] = True
        for param in self.policy.parameters():
            param.data = wp
            break
        for param in self.vf.parameters():
            param.data = wv
            break
        for param in self.qf.parameters():
            param.data = wq
            break
        for i in range(self._num_skills):
            self.batch_env[i].reward_fn.skill_table = skill_table
            # TODO modify this skill setting logic to avoid probleems in future
            self.batch_env[i].reward_fn.set_skill(i)
        self.env.reward_fn.skill_table = skill_table
        self.eval_env.reward_fn.skill_table = skill_table
        self.env.reset()
        self.sampler.reset()

    def on_sanity_check_start(self) -> None:
        if self.single_skill is None:
            self._match_best_skills()
        else:
            self.env.reward_fn.skill_table = np.ones(self._num_skills)*self.single_skill
            self.eval_env.reward_fn.skill_table = np.ones(self._num_skills)*self.single_skill
            self.z = self.get_best_skill(self.policy, self.env, self.hparams.num_skills, self.hparams.max_path_length, self.hparams.num_runs)
            if self.double_skill:
                self.env.reward_fn.skill_table = np.ones(self._num_skills) * self.double_skill
                self.eval_env.reward_fn.skill_table = np.ones(self._num_skills) * self.double_skill
                self.z2 = self.get_best_skill(self.policy, self.env, self.hparams.num_skills,
                                             self.hparams.max_path_length, self.hparams.num_runs)
                for param in self.policy.parameters():
                    wp_ref = param
                    break
                for param in self.vf.parameters():
                    wv_ref = param
                    break
                for param in self.qf.parameters():
                    wq_ref = param
                    break
                wp = wp_ref.clone()
                wv = wv_ref.clone()
                wq = wq_ref.clone()
                wp[:, self.Do - self._num_skills + self.single_skill] = wp_ref[:,
                                                                              self.Do - self._num_skills + self.z]
                wv[:, self.Do - self._num_skills + self.single_skill] = wv_ref[:,
                                                                              self.Do - self._num_skills + self.z]
                wq[:, self.Do - self._num_skills + self.single_skill] = wq_ref[:,
                                                                              self.Do - self._num_skills + self.z]
                print("Allocating skill %d with reward fn %d with retrun %f" % (
                    self.z, self.single_skill, 0))
                wp[:, self.Do - self._num_skills + self.double_skill] = wp_ref[:,
                                                                        self.Do - self._num_skills + self.z2]
                wv[:, self.Do - self._num_skills + self.double_skill] = wv_ref[:,
                                                                        self.Do - self._num_skills + self.z2]
                wq[:, self.Do - self._num_skills + self.double_skill] = wq_ref[:,
                                                                        self.Do - self._num_skills + self.z2]
                print("Allocating skill %d with reward fn %d with retrun %f" % (
                    self.z, self.single_skill, 0))
                for param in self.policy.parameters():
                    param.data = wp
                    break
                for param in self.vf.parameters():
                    param.data = wv
                    break
                for param in self.qf.parameters():
                    param.data = wq
                    break
                self.skills = [self.single_skill,self.double_skill]
                self.z = self.skills[np.random.choice(2)]
                self.env.reward_fn.skill_table = np.arange(self._num_skills)
                self.eval_env.reward_fn.skill_table = np.arange(self._num_skills)
            self.env.reset(state=None, skill=self.z)
            self.eval_env.reset(state=None, skill=self.z)
            self.sampler.reset()
        self.pool.add_samples(self.sampler.sample(self.hparams.min_pool_size, self.policy))
        print("Initialized Replay Buffer with %d samples" % self.pool.size)

    def _split_obs(self, t):
        # TODO: verify that dim is 1, assert shape
        return torch.split(t, [self.Do - self._num_skills, self._num_skills], 1)

    def _sample_z(self):
        """Samples z from p(z), using probabilities in self._p_z."""
        return np.random.choice(self._num_skills, p=self._p_z.detach().cpu().numpy())

    def on_epoch_start(self):
        print("\nCurrent epoch: ",self.current_epoch," Train return for skill ", self.z," with reward fn ", self.env.reward_fn.skill_table[self.z], " :", self.last_path_return)
        if self.single_skill is None:
            self.z = self._sample_z()
        if self.double_skill:
            self.z = self.skills[np.random.choice(2)]
        # self.reward_function.set_skill(self.z)
        self.env.reset(state=None, skill=self.z)
        self.sampler.reset()
        self.eval_env.reset(state=None, skill=self.z)

    def batch_eval(self):
        states = np.zeros((self._num_skills, self.Do))
        next_ob = np.zeros((self._num_skills, self.Do))
        reward = np.zeros(self._num_skills)
        # done = np.zeros(self._num_skills)
        # info = [[] for i in range(self._num_skills)]
        path_return = np.zeros(self._num_skills)
        for i in range(self._num_skills):
            states[i] = self.batch_env[i].reset()
        with self.policy.deterministic(self.hparams.deterministic_eval):
            for j in range(self.hparams.max_path_length):
                actions = self.policy.get_actions(states)
                for i in range(self._num_skills):
                    next_ob[i], reward[i], _, _ = self.batch_env[i].step(actions[i])
                states = next_ob
                path_return += reward
                # TODO handle done for batch env
                # if(done):
                #     break

        return path_return

    def validation_epoch_end(self, outputs) -> OrderedDict:
        state = self.eval_env.reset()
        print(datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y-%m-%d-%H-%M-%S-%f-%Z'))
        # print("Running Validation")
        path_return = 0
        path_length = 0
        self.ims = []
        rewards = [] # storing rewards for skill wise plots
        with self.policy.deterministic(self.hparams.deterministic_eval):
            for i in range(self.hparams.max_path_length):
                action = self.policy.get_actions(state.reshape((1, -1)))
                next_ob, reward, done, info = self.eval_env.step(action)
                if self.hparams.render_validation:
                    # TODO use common resizing everywhere
                    self.ims.append(cv2.resize(self.eval_env.render(mode='rgb_array'), (500, 500)))
                    # print(self.ims[0].shape)#config={'height':500,'width':500,'xpos':0,'ypos':0,'title':'validation'}
                rewards.append(reward)
                state = next_ob
                path_return += reward
                path_length += 1
                if (done):
                    break

        rewards.append(self.current_epoch)
        self.skilldata_val[self.z].append(np.asarray(rewards))
        self.val_path_return = path_return  # TODO : remove printcall back for this, already printed in progress bar
        # Perform batch eval onc eevery n_epochs_eval
        if self.current_epoch % self.hparams.n_epochs_eval == 0 and self.single_skill is None:
            self.batch_return = np.mean(self.batch_eval())

        return OrderedDict({'log': {'path_return': path_return,
                                    'path_length': path_length,
                                    'batch_return': self.batch_return},
                            'progress_bar': {'val_ret': path_return,
                                             'path_len': path_length}})

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

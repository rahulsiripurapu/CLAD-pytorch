import os

import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import dateutil.tz
import datetime
import cv2
import torch
import torch.nn as nn

from algos.diayn import DIAYN
from reward_functions.discriminator import Discriminator

from utils.config import Config

class DIAYN_viz(DIAYN):
    def __init__(self, config: Config):
        super().__init__(config)
        self.num_runs = config.num_runs
        self.ext = config.ext
        if self.ext is None:
            self.ext = ""
        self.path_name = config.path_name
        self.sim_policy = config.sim_policy
        self.skill = config.skill
        self.skill_list = None
        if config.skill_list:
            self.skill_list = np.loadtxt(config.skill_list)
            print("Loaded skill list",self.skill_list)
        self.discriminator = Discriminator(self.Do - self._num_skills, [self.hparams.layer_size, self.hparams.layer_size],
                                       self._num_skills)
        self.distiller = [
            Discriminator(self.Do - self._num_skills, [self.hparams.disc_size[i], self.hparams.disc_size[i]],
                          self._num_skills) for i in range(len(self.hparams.disc_size))]

    # def on_sanity_check_start(self):
    #     self.pool.add_samples(self.sampler.sample(self.hparams.min_pool_size, self.policy))
    #     print("Initialized Replay Buffer with %d samples" % self.pool.size)
    #     ckpt = torch.load("/Users/rahulsiripurapu/CLAD/ckpts-visualize/half-cheetah/DIAYN/seed3-finaldiscriminator/epoch=990.ckpt",map_location="cpu")
    #     self.discriminator= Discriminator(self.Do - self._num_skills, [self.hparams.layer_size, self.hparams.layer_size],
    #                                    self._num_skills)
    #     self.discriminator.load_state_dict(ckpt['state_dict'], strict=False)

    def training_step(self, batch, batch_idx, optimizer_idx) -> OrderedDict:

        states, actions, rewards, dones, next_states = batch

        if self.sim_policy:
            print("Generating video for skill ", self.skill)
            ims,rews,ret,rews2,ret2 = self.run_skill(self.skill)
            _save_video(ims, self.path_name + str(self.skill) + '.mp4')
        else:
            self.get_best_skill(self.policy, self.env, self._num_skills, 1000)
            avg_ret = 0
            avg_ret2 = 0
            for i in range(self._num_skills):
                cur_skill=i
                if self.skill_list is not None:
                    cur_skill=int(self.skill_list[i])
                print("Generating video for skill ",cur_skill)
                ims,rews,ret,rews2,ret2 = self.run_skill(i)
                _save_video(ims, self.path_name + str(cur_skill) + self.ext + '.mp4')
                _save_plot(rews,self.path_name + str(cur_skill) + self.ext+ ".png",rews2)
                avg_ret+=ret
                avg_ret2+=ret2
            print("Average path return: ",avg_ret/self._num_skills)
            if self.hparams.eval_distilled:
                print("Average path return for distilled: ",avg_ret2/self._num_skills)

        return OrderedDict({'loss': path_return})

    def run_skill(self,skill):
        ims = []
        rewards = []
        rewards2 = []
        for j in range(self.num_runs):
            state = self.env.reset(state=None, skill=skill)
            # print(datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y-%m-%d-%H-%M-%S-%f-%Z'))
            # print("Running Validation")
            path_return = 0
            path_return2 = 0
            path_length = 0


            with self.policy.deterministic(self.hparams.deterministic_eval):
                for k in range(self.hparams.max_path_length):
                    action = self.policy.get_actions(state.reshape((1, -1)))
                    next_ob, reward, done, info = self.env.step(action)
                    (obs, z) = self._split_obs(torch.FloatTensor(next_ob)[None, :])
                    if self.on_gpu:
                        # Neeeded because inputs are not on GPU during sample collection
                        # in sanity check TODO: Sanity check is not the place for collecting samples.
                        obs = obs.cuda(self.hparams.device)
                        z = z.cuda(self.hparams.device)
                    logits = self.discriminator(obs)  # N x num_skills
                    skillz = torch.argmax(z, dim=-1)  # N
                    reward = -1 * nn.CrossEntropyLoss(reduction='none')(logits, skillz)  # N
                    reward = torch.clamp(reward, min=-8)
                    assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                    p_z = torch.sum(self._p_z * z, dim=-1)  # N
                    log_p_z = torch.log(p_z + self.hparams.eps)
                    if self.hparams.add_p_z:
                        reward -= log_p_z
                        assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                    if self.hparams.render_validation:
                        # self.env.render(mode="human")
                        ims.append(cv2.resize(self.env.render(mode='rgb_array'),(500,500)))
                        # print(self.ims[0].shape)#config={'height':500,'width':500,'xpos':0,'ypos':0,'title':'validation'}
                        # print(reward)
                    state = next_ob
                    path_return += reward
                    path_length += 1
                    rewards.append(reward)
                    if(self.hparams.eval_distilled):
                        logits = self.distiller(obs)  # N x num_skills
                        reward = -1 * nn.CrossEntropyLoss(reduction='none')(logits, skillz)  # N
                        reward = torch.clamp(reward, min=-8)
                        assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                        p_z = torch.sum(self._p_z * z, dim=-1)  # N
                        log_p_z = torch.log(p_z + self.hparams.eps)
                        if self.hparams.add_p_z:
                            reward -= log_p_z
                            assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
                        path_return2 += reward
                        rewards2.append(reward)
                    if (done):
                        break
            print(path_return)
            print(path_return2)

        return ims, rewards, path_return, rewards2, path_return2

def _save_plot(rews, filename, rews2 = None):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rews)), np.asarray(rews),label="skill reward")
    if rews2 is not None:
        ax.plot(np.arange(len(rews2)), np.asarray(rews2),label="distilled reward")
    ax.set(xlabel="Step", ylabel="Reward")
    ax.set_title("Skill reward over trajectory")
    plt.legend()
    plt.savefig(filename)
    plt.close()

def _save_video(ims, filename):
    # assert all(['ims' in path for path in paths])
    # ims = [im for path in paths for im in path['ims']]
    _make_dir(filename)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)






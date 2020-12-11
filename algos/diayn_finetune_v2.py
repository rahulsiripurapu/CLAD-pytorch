import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
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


from algos.sac_v2 import SAC
from envs.env_selector import env_selector
from envs.option_wrapper import optionwrap
from envs.normalized_env import normalize
from policies.gmm_policy import GMMPolicy
from replay_buffers import SimpleReplayBuffer
from reward_functions.discriminator import Discriminator
from utils.config import Config
from utils.sampler import Sampler
from value_functions.value_function import ValueFunction


class DIAYN_finetune(SAC):
    # Runs SAC but on fixed option env
    # policy, vf, qf, target_vf, must be initialized from checkpoint (Resume from checkpoint works?)
    # Sets z = best skill before running SAC

    def __init__(self, config: Config) -> None:

        super().__init__(config)
        self.z = 0
        self._num_skills = self.hparams.num_skills
        self.discriminator = Discriminator(self.Do - self._num_skills, [config.disc_size, config.disc_size],
                                           self._num_skills)
        self._discriminator_lr = config.lr
        self._p_z = torch.FloatTensor(np.full(self._num_skills, 1.0 / self._num_skills))

    def on_sanity_check_start(self) -> None:
        self.z = self.get_best_skill(self.policy, self.env, self.hparams.num_skills, self.hparams.max_path_length, self.hparams.num_runs)
        self._num_skills = self.hparams.num_skills
        self.env.reset(state=None, skill=self.z)
        self.eval_env.reset(state=None, skill=self.z)
        #TODO sampler reset logic and epoch length interaction seems adhoc
        self.sampler.reset()
        self.pool.add_samples(self.sampler.sample(self.hparams.min_pool_size, self.policy))
        print("Initialized Replay Buffer with %d samples" % self.pool.size)

    # def on_epoch_start(self):
    #     print("Avoiding skill change :",self.global_step," :", self.current_epoch, " :",self.hparams.max_epochs)
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

from envs.env_selector import env_selector
from policies.gmm_policy_h import GMMPolicy
from replay_buffers import SimpleReplayBuffer
from utils.config import Config
from utils.sampler import Sampler
from value_functions.value_function import ValueFunction

EPS = 1E-6


class DISTILL_Q(pl.LightningModule):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hparams = config

        self.env = env_selector(self.hparams)
        self.Da = self.env.action_space.flat_dim
        self.Do = self.env.observation_space.flat_dim
        self.q1 = ValueFunction(self.Do + self.Da, [config.layer_size, config.layer_size])
        self.q2 = ValueFunction(self.Do + self.Da, [config.layer_size, config.layer_size])
        # Constructs a value function mlp with Relu hidden non-linearities, no output non-linearity and with xavier
        # init for weights and zero init for biases.
        self.q1_target = ValueFunction(self.Do + self.Da, [config.layer_size, config.layer_size])
        self.q2_target = ValueFunction(self.Do + self.Da, [config.layer_size, config.layer_size])
        self.stage = None

        self.pool_train = SimpleReplayBuffer(
            env_spec=self.env.spec,
            max_replay_buffer_size=config.max_pool_size,
        )  # create a replay buffer for state+skill and action.

        self.pool_val = SimpleReplayBuffer(
            env_spec=self.env.spec,
            max_replay_buffer_size=config.max_pool_size,
        )

        self.policy = GMMPolicy(
            env_spec=self.env.spec,
            K=config.K,
            hidden_layer_sizes=[config.layer_size, config.layer_size],
            #TODO: pass both q functions to use policy in deterministic mode
            qf=self.q1_target,
            reg=config.reg,
            device=self.hparams.device,
            reparametrization=True
        )  # GMM policy with K mixtures, no reparametrization trick, regularization

        # TODO: add assertion to test qf of policy and qf of model.



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
        self.modules = ["Policy",self.policy,"Q1",self.q1,"Q2",self.q2,"Q1_target",self.q1_target,"Q2_target",self.q2_target]
        # self.z = self.get_best_skill(self.policy, self.env, self.config.num_skills, self.config.max_path_length)
        # self.env.reset(None,self.z)

        # Runs on CPU as Models are transferred to GPU only by trainer which happens after the lightning model init.
        # Also the reason why wandb logger is not available

    def on_sanity_check_start(self) -> None:
        q1 = ValueFunction(self.Do + self.Da, [self.hparams.layer_size, self.hparams.layer_size])
        self.q1.load_state_dict(q1.state_dict())
        q2 = ValueFunction(self.Do + self.Da, [self.hparams.layer_size, self.hparams.layer_size])
        self.q2.load_state_dict(q2.state_dict())

    def on_epoch_start(self) -> None:
        print("Distilling epoch %d"%self.current_epoch)

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.pool_train, self.hparams.epoch_length, self.hparams.batch_size)
        # TODO: figure out why referencee codeee uses episode length abovee instead of batch size

        def _init_fn(worker_id):
            np.random.seed(self.hparams.seed + worker_id)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers,
                                worker_init_fn=_init_fn
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.pool_val, self.hparams.epoch_length, self.hparams.batch_size)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                # num_workers=5
                                )
        return dataloader

    def training_step(self, batch, batch_idx) -> OrderedDict:

        states, actions, rewards, dones, next_states = batch

        self.q1_values = self.q1(states, actions)
        self.q2_values = self.q2(states, actions)
        # assert (self.policy._qf(states,actions)==self.q_values).all()
        with torch.no_grad():
            q1_next_target = self.q1_target(states, actions)  # N
            q2_next_target = self.q2_target(states, actions)


        self._td1_loss = torch.mean((q1_next_target - self.q1_values) ** 2)
        self._td2_loss = torch.mean((q2_next_target - self.q2_values) ** 2)

        return OrderedDict(
            {'loss': self._td1_loss + self._td2_loss,
             'log': {'qf_distill_loss_%d'%self.stage: self._td1_loss + self._td2_loss,
                     'qf_distill_value_%d'%self.stage: torch.mean(self.q1_values)},
             'progress_bar': {'qf_loss': self._td1_loss + self._td2_loss,
                              'qf_mu': torch.mean(self.q1_values)}})

        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizers = []
        optimizers.append(optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters())
                                     , lr=self._qf_lr))
        return optimizers

    def forward(self, *args, **kwargs):
        return None

    def validation_step(self, batch, batch_idx) -> OrderedDict:
        states, actions, rewards, dones, next_states = batch
        q1_values = self.q1(states, actions)
        q2_values = self.q2(states, actions)
        # assert (self.policy._qf(states,actions)==self.q_values).all()
        with torch.no_grad():
            q1_next_target = self.q1_target(states, actions)  # N
            q2_next_target = self.q2_target(states, actions)

        td1_loss = torch.mean((q1_next_target - q1_values) ** 2)
        td2_loss = torch.mean((q2_next_target - q2_values) ** 2)

        return OrderedDict({'val_loss': td2_loss + td1_loss})

    def validation_epoch_end(self,outputs) -> OrderedDict:
        # called at the end of a validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss_distill_qf_%d'%self.stage: avg_loss}
        return {'val_loss': avg_loss, 'log': log}



class DISTILL_P(pl.LightningModule):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hparams = config

        self.env = env_selector(self.hparams)
        self.Da = self.env.action_space.flat_dim
        self.Do = self.env.observation_space.flat_dim
        self.q1 = ValueFunction(self.Do + self.Da, [config.layer_size, config.layer_size])
        self.q2 = ValueFunction(self.Do + self.Da, [config.layer_size, config.layer_size])
        # Constructs a value function mlp with Relu hidden non-linearities, no output non-linearity and with xavier
        # init for weights and zero init for biases.
        self.q1_target = ValueFunction(self.Do + self.Da, [config.layer_size, config.layer_size])
        self.q2_target = ValueFunction(self.Do + self.Da, [config.layer_size, config.layer_size])

        self.pool_train = SimpleReplayBuffer(
            env_spec=self.env.spec,
            max_replay_buffer_size=config.max_pool_size,
        )  # create a replay buffer for state+skill and action.

        self.pool_val = SimpleReplayBuffer(
            env_spec=self.env.spec,
            max_replay_buffer_size=config.max_pool_size,
        )
        self.policy = GMMPolicy(
            env_spec=self.env.spec,
            K=config.K,
            hidden_layer_sizes=[config.layer_size, config.layer_size],
            # TODO: pass both q functions to use policy in deterministic mode
            qf=self.q1_target,
            reg=config.reg,
            device=self.hparams.device,
            reparametrization=True
        )  # GMM policy with K mixtures, no reparametrization trick, regularization

        # TODO: add assertion to test qf of policy and qf of model.

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
        self.modules = ["Policy", self.policy, "Q1", self.q1, "Q2", self.q2, "Q1_target", self.q1_target, "Q2_target",
                        self.q2_target]
        # self.z = self.get_best_skill(self.policy, self.env, self.config.num_skills, self.config.max_path_length)
        # self.env.reset(None,self.z)

        # Runs on CPU as Models are transferred to GPU only by trainer which happens after the lightning model init.
        # Also the reason why wandb logger is not available

    def on_sanity_check_start(self) -> None:
        self.policy_target = GMMPolicy(
            env_spec=self.env.spec,
            K=self.hparams.K,
            hidden_layer_sizes=[self.hparams.layer_size, self.hparams.layer_size],
            # TODO: pass both q functions to use policy in deterministic mode
            qf=self.q1_target,
            reg=self.hparams.reg,
            device=self.hparams.device,
            reparametrization=True
        )
        self.policy_target.load_state_dict(self.policy.state_dict())
        policy = GMMPolicy(
            env_spec=self.env.spec,
            K=self.hparams.K,
            hidden_layer_sizes=[self.hparams.layer_size, self.hparams.layer_size],
            # TODO: pass both q functions to use policy in deterministic mode
            qf=self.q1_target,
            reg=self.hparams.reg,
            device=self.hparams.device,
            reparametrization=True
        )
        self.policy.load_state_dict(policy.state_dict())
        print("\n\n\nIs self on gpu???",self.on_gpu)
        if self.on_gpu:
            self.policy_target = self.policy_target.cuda(self.hparams.device)
            print("Moving policy_target to GPU")

    def on_epoch_start(self) -> None:
        print("Distilling epoch %d"%self.current_epoch)

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.pool_train, self.hparams.epoch_length, self.hparams.batch_size)

        # TODO: figure out why referencee codeee uses episode length abovee instead of batch size

        def _init_fn(worker_id):
            np.random.seed(self.hparams.seed + worker_id)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers,
                                worker_init_fn=_init_fn
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.pool_val, self.hparams.epoch_length, self.hparams.batch_size)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                # num_workers=5
                                )
        return dataloader

    def training_step(self, batch, batch_idx) -> OrderedDict:
        states, actions, rewards, dones, next_states = batch

        distributions, action_samples, log_probs, corr, reg_loss = self.policy(states)
        # print(log_probs.shape)
        # assert log_probs.shape == torch.Size([action_samples.shape[0]])
        # values1 = self.q1(states, action_samples)
        # values2 = self.q2(states, action_samples)
        # self.value = torch.min(values1, values2)  # N
        # # print(action_samples.shape,log_probs.shape,reg_loss.shape,states.shape) #TODO assert shapes
        #
        # # with torch.no_grad():
        # # TODO : check grad
        # self.scaled_log_pi = self._scale_entropy * (log_probs - corr)
        # self._policy_loss = torch.mean(self.scaled_log_pi - self.value)

        with torch.no_grad():
            target_dist, _, _, _, _ = self.policy_target(states)  # N

        self._policy_loss = torch.mean((target_dist[0] - distributions[0]) ** 2) \
                            + torch.mean((target_dist[1] - distributions[1]) ** 2) \
                            + torch.mean((target_dist[2] - distributions[2]) ** 2)

        log = {
               'policy_loss_distill_%d'%self.stage: self._policy_loss,
               }
        status = {
            'train_loss': self._policy_loss,
        }

        return OrderedDict({'loss': self._policy_loss,
                            'log': log, 'progress_bar': status})
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizers = []
        optimizers.append(optim.Adam(self.policy.parameters(), lr=self._policy_lr))
        return optimizers

    def forward(self, *args, **kwargs):
        return None

    def validation_step(self, batch, batch_idx) -> OrderedDict:
        states, actions, rewards, dones, next_states = batch
        distributions, action_samples, log_probs, corr, reg_loss = self.policy(states)
        # print(log_probs.shape)
        # assert log_probs.shape == torch.Size([action_samples.shape[0]])
        # values1 = self.q1(states, action_samples)
        # values2 = self.q2(states, action_samples)
        # self.value = torch.min(values1, values2)  # N
        # # print(action_samples.shape,log_probs.shape,reg_loss.shape,states.shape) #TODO assert shapes
        #
        # # with torch.no_grad():
        # # TODO : check grad
        # self.scaled_log_pi = self._scale_entropy * (log_probs - corr)
        # self._policy_loss = torch.mean(self.scaled_log_pi - self.value)

        with torch.no_grad():
            target_dist, _, _, _, _ = self.policy_target(states)  # N

        policy_loss = torch.mean((target_dist[0] - distributions[0]) ** 2) \
                            + torch.mean((target_dist[1] - distributions[1]) ** 2) \
                            + torch.mean((target_dist[2] - distributions[2]) ** 2)

        return OrderedDict({'val_loss': policy_loss})

    def validation_epoch_end(self,outputs) -> OrderedDict:
        # called at the end of a validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss_distill_policy_%d'%self.stage: avg_loss}
        return {'val_loss': avg_loss, 'log': log}

class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: SimpleReplayBuffer, epoch_length: int = 1000, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size
        self.epoch_length = epoch_length

    def __len__(self) -> int:
        return self.epoch_length

    def __iter__(self) -> Tuple:
        # batch = self.buffer.random_batch(self.sample_size)
        # for i in range(len(batch['terminals'])):
        #TODO: divide by num_workers to get correct epoch length.
        for j in range(self.epoch_length):
            # print("\nfetching new batch: ",j,self.buffer.size)
            batch = self.buffer.random_batch(self.sample_size)
            # print(np.mean(batch['rewards']))
            for i in range(len(batch['dones'])):
                yield batch['observations'][i], batch['actions'][i], batch['rewards'][i], batch['dones'][i], \
                      batch['next_observations'][i]

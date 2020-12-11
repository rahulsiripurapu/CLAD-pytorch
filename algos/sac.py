import pytorch_lightning as pl
from collections import OrderedDict, deque
from typing import Tuple, List
import numpy as np
import dateutil.tz
import datetime
import gc

import cv2
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from envs.env_selector import env_selector
from policies.gmm_policy import GMMPolicy
from replay_buffers import SimpleReplayBuffer
from utils.config import Config
from utils.sampler import Sampler
from value_functions.value_function import ValueFunction

EPS = 1E-6


class SAC(pl.LightningModule):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hparams = config

        self.env = env_selector(self.hparams)  # TODO: normalization is not required but will it be needed?
        self.eval_env = env_selector(self.hparams, config.seed + 1)
        self.Da = self.env.action_space.flat_dim
        self.Do = self.env.observation_space.flat_dim  # includes skill in case env is option wrapped
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
            device=self.hparams.device
        )  # GMM policy with K mixtures, no reparametrization trick, regularization
        self.modules = ["Policy", self.policy, "QF", self.qf, "VF", self.vf, "VF_Target", self.vf_target]

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
        # Runs on CPU(moved sampling to (on_train_start) to avoid bug in DIAYN + use GPU instead of CPU(No need for device logic!!) as Models are transferred to GPU only by trainer which happens after the lightning model init.
        # TODO remove device logic in Policy
        # Also the reason why wandb logger is not available
        self.batch_idx = None
        # torch.autograd.set_detect_anomaly(True) #TODO: disable if compute overhead

    def get_best_skill(self, policy, env, num_skills, max_path_length, n_paths=1):
        print('Finding best skill...')
        reward_list = []
        with policy.deterministic(self.hparams.deterministic_eval):
            for z in range(num_skills):
                env.reset(state=None, skill=z)
                total_returns = 0
                sampler = Sampler(env, max_path_length)
                for p in range(n_paths):
                    new_paths = sampler.sample(max_path_length, policy)
                    total_returns += new_paths[-1]['path_return']
                print('Reward for skill %d = %.3f' % (z, total_returns))
                reward_list.append(total_returns)

        best_z = np.argmax(reward_list)
        print('Best skill found: z = %d, reward = %d, seed = %d' % (best_z,
                                                                    reward_list[best_z], self.hparams.seed))
        return best_z

    def on_sanity_check_start(self) -> None:
        self.pool.add_samples(self.sampler.sample(self.hparams.min_pool_size, self.policy))
        print("Initialized Replay Buffer with %d samples" % self.pool.size)

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.pool, self.hparams.epoch_length, self.hparams.batch_size)

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
        dataset = RLDataset(self.pool, 1, 1)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=1,
                                # num_workers=5
                                )
        return dataloader

    # def _split_obs(self,t):
    # TODO remove from DIAYN, herf, and v2?
    #     # TODO: verify that dim is 1, assert shape
    #     return torch.split(t, [self._Do, self._num_skills], 1)

    def training_step(self, batch, batch_idx, optimizer_idx) -> OrderedDict:

        states, actions, rewards, dones, next_states = batch
        self.batch_idx = batch_idx
        # print(states[0], batch_idx)

        # print(self.pool.size,optimizer_idx,batch_idx,states[0])
        # print("Running train",states.shape,batch_idx,optimizer_idx)

        # TODO: vars are already floatTensors.
        # Train Policy
        if optimizer_idx == 0:
            # for param in self.policy.parameters():
            #     print(param.names, param.size(), param.requires_grad)
            # print("Done")
            # for param in self.vf.parameters():
            #     print(param.names, param.size(), param.requires_grad)
            # print("Donevf")
            # print(torch.max(rewards),torch.min(rewards),torch.mean(rewards))
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

            # How is this kl surrogate loss derived?
            self._kl_surrogate_loss = torch.mean(log_probs * (
                    self.scaled_log_pi - self.log_targets + self.values.detach()))
            self._policy_loss = reg_loss + self._kl_surrogate_loss
            self._vf_loss = 0.5 * torch.mean(
                (self.values - self.log_targets + self.scaled_log_pi) ** 2)

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
                'train_loss': self._policy_loss.detach().cpu().numpy(),
                # 'vf_loss': self._vf_loss,
                # 'steps': torch.tensor(self.global_step),#.to(device),#Where did this global_step comee from is it PL inbuilt?
                'max_ret': self.max_path_return,
                'last_ret': self.last_path_return,
                'gmm_mu': torch.mean(distributions.component_distribution.mean).detach().cpu().numpy(),
                'gmm_sig': torch.mean(distributions.component_distribution.stddev).detach().cpu().numpy(),
                'vf_loss': self._vf_loss.detach().cpu().numpy(),
                'vf_mu': torch.mean(self.values).detach().cpu().numpy()
            }

            return OrderedDict({'loss': self._policy_loss + self._vf_loss,
                                'log': log, 'progress_bar': status})

        # TODO is it faster if qf is also optimized simultaneously along with vf and policy?

        # Train QF
        if optimizer_idx == 1:
            # for param in self.qf.parameters():
            #     print(param.names, param.size(), param.requires_grad)
            # print("Doneqf")
            self.q_values = self.qf(states, actions)
            # assert (self.policy._qf(states,actions)==self.q_values).all()
            with torch.no_grad():
                vf_next_target = self.vf_target(next_states)  # N
                ys = self._scale_reward * rewards + (1 - dones) * self._discount * vf_next_target  # N

            self._td_loss = 0.5 * torch.mean((ys - self.q_values) ** 2)

            return OrderedDict(
                {'loss': self._td_loss,
                 'log': {'qf_loss': self._td_loss.detach().cpu().numpy(),
                         'qf_value': torch.mean(self.q_values).detach().cpu().numpy(),
                         'rewards': torch.mean(rewards).detach().cpu().numpy()},
                 'progress_bar': {'qf_loss': self._td_loss,
                                  'rewards': torch.mean(rewards).detach().cpu().numpy(),
                                  'qf_mu': torch.mean(self.q_values).detach().cpu().numpy()}})

        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)

    def on_batch_end(self) -> None:
        with torch.no_grad():
            for vf, vf_targ in zip(self.vf.parameters(), self.vf_target.parameters()):
                vf_targ.data.mul_(1 - self.hparams.tau)
                vf_targ.data.add_(self.hparams.tau * vf.data)

    def validation_step(self, batch, batch_idx) -> OrderedDict:
        # state = self.eval_env.reset()
        # print("Running Validation step")
        # path_return = 0
        # path_length = 0
        # for i in range(self.config.max_path_length):
        #     action = self.policy.get_actions(state.reshape((1, -1)))
        #     next_ob, reward, terminal, info = self.env.step(action)
        #     state = next_ob
        #     path_return += reward
        #     path_length += 1
        #     if(terminal):
        #         break

        return OrderedDict({'val_ret': 0, 'path_len': 0})

    def validation_epoch_end(self, outputs) -> OrderedDict:
        gc.collect()
        state = self.eval_env.reset()
        print(datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y-%m-%d-%H-%M-%S-%f-%Z'))
        # print("Running Validation")
        path_return = 0
        path_length = 0
        self.ims = []
        with self.policy.deterministic(self.hparams.deterministic_eval):
            # TODO add support for n_eval_iters
            for i in range(self.hparams.max_path_length):
                action = self.policy.get_actions(state.reshape((1, -1)))
                next_ob, reward, done, info = self.eval_env.step(action)
                if self.hparams.render_validation:
                    # TODO use common resizing everywhere
                    self.ims.append(cv2.resize(self.eval_env.render(mode='rgb_array'), (500, 500)))
                    # print(self.ims[0].shape)#config={'height':500,'width':500,'xpos':0,'ypos':0,'title':'validation'}
                state = next_ob
                path_return += reward
                path_length += 1
                if done:
                    break

        self.val_path_return = path_return  # TODO : remove printcall back for this, already printed in progress bar
        return OrderedDict({'log': {'path_return': path_return,
                                    'path_length': path_length},
                            'progress_bar': {'val_ret': path_return,
                                             'path_len': path_length}})

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizers = []
        # TODO: combining vf and policy, figure out more elegant way to have unlinked learning rates than as
        # a multiplication factor in the loss sum. Also figure out why having them separate doesn't increase
        # compute time by the expected
        optimizers.append(optim.Adam(list(self.policy.parameters()) + list(self.vf.parameters())
                                     , lr=self._policy_lr))
        # optimizers.append(optim.Adam(self.vf.parameters(), lr=self._vf_lr))
        optimizers.append(optim.Adam(self.qf.parameters(), lr=self._qf_lr))
        return optimizers

    def forward(self, *args, **kwargs):
        return None

    def check_modules(self):
        self.policy.cuda(self.hparams.device)
        self.vf.cuda(self.hparams.device)
        self.qf.cuda(self.hparams.device)
        self.vf_target.cuda(self.hparams.device)
        for param in self.policy.parameters():
            print(param.data.shape, param.data.mean(), param.data.max(), param.data.min(), param.data.std())
        for param in self.vf.parameters():
            print(param.data.shape, param.data.mean(), param.data.max(), param.data.min(), param.data.std())
        for param in self.qf.parameters():
            print(param.data.shape, param.data.mean(), param.data.max(), param.data.min(), param.data.std())
        for param in self.vf_target.parameters():
            print(param.data.shape, param.data.mean(), param.data.max(), param.data.min(), param.data.std())


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
        # TODO: divide by num_workers to get correct epoch length.
        for j in range(self.epoch_length):
            # print("\nfetching new batch: ",j,self.buffer.size)
            batch = self.buffer.random_batch(self.sample_size)
            # print(np.mean(batch['rewards']))
            for i in range(len(batch['dones'])):
                yield batch['observations'][i], batch['actions'][i], batch['rewards'][i], batch['dones'][i], \
                      batch['next_observations'][i]

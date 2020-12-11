import numpy as np

from utils.serializable import Serializable

from .replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, env_spec, max_replay_buffer_size):
        Serializable.quick_init(self, locals())
        super(SimpleReplayBuffer, self).__init__(env_spec)

        max_replay_buffer_size = int(max_replay_buffer_size)
        # TODO: add support for changing buffer datatypes and attributes

        self._env_spec = env_spec
        self._observation_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim),dtype='float32')
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        # TODO: add logic for termination condition to save memory
        # TODO: at least convert skill to int or use sparse matrix to save mem
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim),dtype='float32')
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim),dtype='float32')
        self._rewards = np.zeros(max_replay_buffer_size,dtype='float32')
        # self._terminals[i] = a terminal was received at time i
        self._dones = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, obs, action, reward, done,
                   next_obs):
        #TODO assert order of arguments is correct
        self._observations[self._top] = obs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._dones[self._top] = done
        self._next_obs[self._top] = next_obs

        self._advance()

    def add_samples(self,samples):
        for i in range(len(samples)):
            sample = samples[i]
            self.add_sample(sample['obs'], sample['action'], sample['reward'],
                            sample['done'], sample['next_obs'])

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            dones=self._dones[indices],
            next_observations=self._next_obs[indices],
        )

    @property
    def size(self):
        return self._size

    def __getstate__(self):
        d = super(SimpleReplayBuffer, self).__getstate__()
        d['__args'] += (self._max_buffer_size,)
        d.update(dict(
            o=self._observations.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._dones.tobytes(),
            no=self._next_obs.tobytes(),
            top=self._top,
            size=self._size,
        ))
        return d

    def __setstate__(self, d):
        super(SimpleReplayBuffer, self).__setstate__(d)
        self._observations = np.fromstring(d['o'], dtype='float32').reshape(
            self._max_buffer_size, -1
        )
        self._next_obs = np.fromstring(d['no'], dtype='float32').reshape(
            self._max_buffer_size, -1
        )
        self._actions = np.fromstring(d['a'], dtype='float32').reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r'], dtype='float32').reshape(self._max_buffer_size)
        self._dones = np.fromstring(d['t'], dtype=np.uint8)
        self._top = d['top']
        self._size = d['size']

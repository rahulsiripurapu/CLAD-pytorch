# Code from learning_to_adapt

import numpy as np

import spaces
from envs.env_spec import EnvSpec
from utils.serializable import Serializable
from gym.spaces import Box

# from rand_param_envs.gym.spaces import Box as OldBox
# TODO: Serializable is needed as lightning module doesn't save envs automatically to ckpt.
"""
Fixes an option to the environment class.

Args:
    EnvCls (gym.Env): class of the unnormalized gym environment
    env_args (dict or None): arguments of the environment

Returns:
    Fixed Option environment

"""


class OptionWrapper(Serializable):
    """
    Adds a fixed option to the environment class.

    Args:
        Env (gym.Env): class of the gym environment
        num_skills: number of skills to produce one hot vectors for
        skill: fixed skill of the environment
    """

    def __init__(self,
                 env,
                 num_skills: int = 50,
                 skill: int = 0,
                 reward_fn=None
                 ):
        Serializable.quick_init(self, locals())

        self._wrapped_env = env
        self.num_skills = num_skills
        self.z = skill
        self.reward_fn = reward_fn
        obs_space = self.observation_space
        low = np.hstack([obs_space.low, np.full(num_skills, 0)])
        high = np.hstack([obs_space.high, np.full(num_skills, 1)])
        self.observation_space = spaces.Box(low=low, high=high)
        # Calls inherited action_space function and assigns
        self.action_space = self.action_space
        self.spec = EnvSpec(self.observation_space, self.action_space)
        self.state = None
        self.reset(state=None, skill=self.z)

    # @property
    # def action_space(self):
    #     if isinstance(self._wrapped_env.action_space, Box):
    #         ub = np.ones(self._wrapped_env.action_space.shape) * self._normalization_scale
    #         return Box(-1 * ub, ub, dtype=np.float32)
    #     return self._wrapped_env.action_space

    @staticmethod
    def concat_obs_z(obs, z, num_skills):
        """Concatenates the observation to a one-hot encoding of Z."""
        assert np.isscalar(z)
        z_one_hot = np.zeros(num_skills)
        z_one_hot[z] = 1
        return np.hstack([obs, z_one_hot])

    def set_reward_fn(self, reward_fn):
        self.reward_fn = reward_fn
        self.reward_fn.set_skill(self.z) # Sets skill to measure in reward function.
        # Skill value here may not correspond to fixed rewards in reward_fn, skill_table used to reassign.

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        orig_attr = self._wrapped_env.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr

    def reset(self, state=None, skill=None):
        if skill is not None:
            self.z = skill
            if self.reward_fn:
                self.reward_fn.set_skill(self.z)
            # print("Resetting skill: ", self.z)
        obs = self._wrapped_env.reset(state)
        self.state = self.concat_obs_z(obs, self.z, self.num_skills)
        return self.state

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["num_skills"] = self.num_skills
        d["z"] = self.z
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.num_skills = d["num_skills"]
        self.z = d["z"]

    def step(self, action):

        if self.reward_fn:
            o1 = self.env.model.data.qpos.flatten()
        (obs, r, done, info) = self._wrapped_env.step(action)
        if self.reward_fn:
            o2 = self.env.model.data.qpos.flatten()
            r = self.reward_fn.get_reward(o1, o2, action, self.env.dt)

        aug_obs = self.concat_obs_z(obs, self.z, self.num_skills)
        self.state = aug_obs
        return aug_obs, r, done, info

    def m_step(self, action):
        """Computes a vector of rewards for all available reward functions"""
        o1 = self.env.model.data.qpos.flatten()
        (obs, r, done, info) = self._wrapped_env.step(action)
        o2 = self.env.model.data.qpos.flatten()
        r = self.reward_fn.get_reward_vec(o1, o2, action, self.env.dt, self.num_skills)
        aug_obs = self.concat_obs_z(obs, self.z, self.num_skills)
        self.state = aug_obs
        return aug_obs, r, done, info


optionwrap = OptionWrapper

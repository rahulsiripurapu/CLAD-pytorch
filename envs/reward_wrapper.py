import numpy as np

import spaces
from envs.env_spec import EnvSpec
from utils.serializable import Serializable

# TODO: Serializable is needed as lightning module doesn't save envs automatically to ckpt.
"""
Fixes an option to the environment class.

Args:
    EnvCls (gym.Env): class of the unnormalized gym environment
    env_args (dict or None): arguments of the environment

Returns:
    Fixed Option environment

"""


class RewardWrapper(Serializable):
    """
    Adds a fixed option to the environment class.

    Args:
        Env (gym.Env): class of the gym environment
        num_skills: number of skills to produce one hot vectors for
        skill: fixed skill of the environment
    """

    def __init__(self,
                 env,
                 reward_fn
                 ):
        Serializable.quick_init(self, locals())

        self._wrapped_env = env
        self.reward_function = reward_fn

        self.num_skills = num_skills
        self.z = skill
        obs_space = self.observation_space
        low = np.hstack([obs_space.low, np.full(num_skills, 0)])
        high = np.hstack([obs_space.high, np.full(num_skills, 1)])
        self.observation_space = spaces.Box(low=low, high=high)
        # TODO verfiy below line is doing something  useful
        self.action_space = self.action_space
        print("Veriifying action space: ",self.action_space)
        self.spec = EnvSpec(self.observation_space, self.action_space)
        self.reset(state=None,skill=self.z)

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

    # def reset(self, state=None, skill=None):
    #     if skill is not None:
    #         self.z = skill
    #         print("Resetting skill: ", self.z)
    #     obs = self._wrapped_env.reset(state)
    #     self.state = self.concat_obs_z(obs, self.z, self.num_skills)
    #     return self.state
    #
    # def __getstate__(self):
    #     d = Serializable.__getstate__(self)
    #     d["num_skills"] = self.num_skills
    #     d["z"] = self.z
    #     return d
    #
    # def __setstate__(self, d):
    #     Serializable.__setstate__(self, d)
    #     self.num_skills = d["num_skills"]
    #     self.z = d["z"]

    def step(self, action):

        (obs, r, done, info) = self._wrapped_env.step(action)
        logits = self.reward_function(obs)
        skill = torch.argmax(z, dim=1)  # N
        reward = -1 * nn.CrossEntropyLoss(reduction='none')(logits, skill)  # N
        assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
        p_z = torch.sum(self._p_z * z, dim=1)  # N
        log_p_z = torch.log(p_z + self.hparams.eps)
        if self.hparams.add_p_z:
            reward -= log_p_z
            assert not torch.isnan(reward).any() and not torch.isinf(reward).any()
        return aug_obs, r, done, info


fixoption = RewardWrapper



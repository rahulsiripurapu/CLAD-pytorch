import numpy as np
import os
from utils.serializable import Serializable
from envs.base import Step
from envs.mujoco_env import MujocoEnv



def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahEnv(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets", "half_cheetah.xml"))
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        print(self.model.data.qpos.flatten().shape,self.model.data.qpos.flatten()[:],
            np.asarray(self.model.data.qvel.flat))

        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            # self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        run_cost = -1 * self.get_body_comvel("torso")[0]
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done)

    # TODO: figure out how to add hparams attribute to model as hparams are automatically logged.
    #  (Solution: pass hparams as part of the config dictionary without classes or using str_to_class()
    # @classmethod
    # def __repr__(self):
    #     return self.__getstate__().__str__()

    # TODO: add logging
    # @overrides
    # def log_diagnostics(self, paths):
    #     progs = [
    #         path["observations"][-1][-3] - path["observations"][0][-3]
    #         for path in paths
    #     ]
    #     logger.record_tabular('AverageForwardProgress', np.mean(progs))
    #     logger.record_tabular('MaxForwardProgress', np.max(progs))
    #     logger.record_tabular('MinForwardProgress', np.min(progs))
    #     logger.record_tabular('StdForwardProgress', np.std(progs))

if __name__ == '__main__':
    env = HalfCheetahEnv()
    while True:
        state = env.reset()
        print(state)
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render(mode='human')

#TODO add script to test reproducibility, action space range violation effects and reset capability of env
#
# if __name__ == '__main__':
#     env = HalfCheetahEnv()
#     ref = env.reset()
#     action = np.ones((1,6))
#     print(action)
#     while True:
#         env.reset(ref)
#         for _ in range(3):
#             next_ob, reward, terminal, info = env.step(action)
#             print(next_ob)
#             env.render()
#         env.reset(ref)
#         action = action +1 # pushing action out of range (-1,1)
#         print(action)
#         for _ in range(3):
#             next_ob, reward, terminal, info = env.step(action)
#             print(next_ob)
#             env.render()
#         break
import numpy as np
import os
from utils.serializable import Serializable
from envs.base import Step
from envs.mujoco_env import MujocoEnv
# from misc import autoargs


class InvertedDoublePendulumEnv(MujocoEnv, Serializable):
    FILE = 'inverted_double_pendulum.xml.mako'

    # TODO: figure out purpose of autoargs
    # @autoargs.arg("random_start", type=bool,
    #               help="Randomized starting position by adjusting the angles"
    #                    "When this is false, the double pendulum started out"
    #                    "in balanced position")
    def __init__(
            self,
            *args, **kwargs):
        self.random_start = kwargs.get("random_start", True)
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets", "inverted_double_pendulum.xml"))
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1],  # cart x pos
            np.sin(self.model.data.qpos[1:]),  # link angles
            np.cos(self.model.data.qpos[1:]),
            np.clip(self.model.data.qvel, -10, 10),
            np.clip(self.model.data.qfrc_constraint, -10, 10)
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        x, _, y = self.model.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.model.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = float(alive_bonus - dist_penalty - vel_penalty)
        done = y <= 1
        return Step(next_obs, r, done)

    def reset_mujoco(self, init_state=None):
        assert init_state is None
        qpos = np.copy(self.init_qpos)
        if self.random_start:
            qpos[1] = (np.random.rand() - 0.5) * 40 / 180. * np.pi
        self.model.data.qpos = qpos
        self.model.data.qvel = self.init_qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl

if __name__ == '__main__':
    env = InvertedDoublePendulumEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()
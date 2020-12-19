import numpy as np

""" HC state space:
source: https://github.com/openai/gym/blob/1d8565717206e54fca9e73ea1b3969948b464c3c/gym/envs/mujoco/assets/half_cheetah.xml
State-Space (name/joint/parameter):
        - rootx     slider      position (m)
        - rootz     slider      position (m)
        - rooty     hinge       angle (rad)
        - bthigh    hinge       angle (rad)
        - bshin     hinge       angle (rad)
        - bfoot     hinge       angle (rad)
        - fthigh    hinge       angle (rad)
        - fshin     hinge       angle (rad)
        - ffoot     hinge       angle (rad)
        - rootx     slider      velocity (m/s)
        - rootz     slider      velocity (m/s)
        - rooty     hinge       angular velocity (rad/s)
        - bthigh    hinge       angular velocity (rad/s)
        - bshin     hinge       angular velocity (rad/s)
        - bfoot     hinge       angular velocity (rad/s)
        - fthigh    hinge       angular velocity (rad/s)
        - fshin     hinge       angular velocity (rad/s)
        - ffoot     hinge       angular velocity (rad/s)
    Actuators (name/actuator/parameter):
        - bthigh    hinge       torque (N m)
        - bshin     hinge       torque (N m)
        - bfoot     hinge       torque (N m)
        - fthigh    hinge       torque (N m)
        - fshin     hinge       torque (N m)
        - ffoot     hinge       torque (N m)


"""


class HERF():
    def __init__(self, num_skills=50):
        self.skill = None
        self.num_skills = num_skills
        self.max_skills = 25  # Currently only supports max skills 25.
        self.skill_table = np.arange(num_skills)
        # default skill table with identity allocation

    def set_skill(self, skill):
        # Uses modulo on skill to support num_skills
        # Each reward function is allocated to index corresponding to its option.
        # skill is the option and the skill_table value is the reward function
        self.skill = self.skill_table[skill] % self.max_skills

    def hill_function(self, val, target):
        # hill function with peak at target and slope 1/-1.
        return np.sign(target) * val if np.sign(target) * (target - val) > 0 else \
            np.abs(target) - np.sign(target) * (val - target)

    def run(self, o1, o2, action, dt, target):
        # Reward function for running with target = target speed for max reward.
        reward_ctrl = - 0.1 * np.square(action).sum()  # control penalty
        vel = (o2[0] - o1[0]) / dt  # derivative of x position of halfcheetah
        reward_run = self.hill_function(vel, target)
        return reward_ctrl + reward_run

    def flip(self, o1, o2, action, dt, target):
        # Reward function for flipping continuously, with target = target angular velocity for max reward
        reward_ctrl = - 0.1 * np.square(action).sum()
        vel = (o2[2] - o1[2]) / dt  # derivative of angular position of hc
        reward_flip = self.hill_function(vel, target)
        return 5*(reward_ctrl + reward_flip)

    def stand(self, o1, o2, action, dt, target):
        # head stand or back stand depending on angle.
        reward_ctrl = - 0.1 * np.square(action).sum()
        angle = o2[2]
        reward_stand = self.hill_function(angle, target)
        return reward_ctrl + reward_stand

    def jump(self, o1, o2, action, dt, target):
        # Jumping while maintaining a stand
        vel = (o2[1] - o1[1]) / dt # velocity in z direction, tries to keep jumping
        return np.abs(vel) + self.stand(o1, o2, action, dt, target)

    def get_reward(self, o1, o2, action, dt):
        assert self.skill is not None # ensures skill is set before calling
        # Running
        if self.skill == 0:
            # Run at speed 8 m/s
            return self.run(o1, o2, action, dt, 8)
        if self.skill == 1:
            return self.run(o1, o2, action, dt, 6)
        if self.skill == 2:
            return self.run(o1, o2, action, dt, 4)
        if self.skill == 3:
            return self.run(o1, o2, action, dt, 2)
        if self.skill == 4:
            return self.run(o1, o2, action, dt, -2)
        if self.skill == 5:
            return self.run(o1, o2, action, dt, -4)
        if self.skill == 6:
            return self.run(o1, o2, action, dt, -6)
        if self.skill == 7:
            return self.run(o1, o2, action, dt, -8)
        # flipping
        if self.skill == 8:
            # Flip with speed 8 rads/sec
            return self.flip(o1, o2, action, dt, 8)
        if self.skill == 9:
            return self.flip(o1, o2, action, dt, 6)
        if self.skill == 10:
            return self.flip(o1, o2, action, dt, 4)
        if self.skill == 11:
            return self.flip(o1, o2, action, dt, 2)
        if self.skill == 12:
            return self.flip(o1, o2, action, dt, -8)
        if self.skill == 13:
            return self.flip(o1, o2, action, dt, -6)
        if self.skill == 14:
            return self.flip(o1, o2, action, dt, -4)
        if self.skill == 15:
            return self.flip(o1, o2, action, dt, -2)
        # Stands
        if self.skill == 16:
            # head stand at an angle ~ pi/2 i.e. 90 degrees
            return self.stand(o1, o2, action, dt, 1.6)
        if self.skill == 17:
            # stand at an angle ~ pi/3 i.e. 60 degrees
            return self.stand(o1, o2, action, dt, 1.0)
        if self.skill == 18:
            # 2pi/3 i.e. 120 degrees
            return self.stand(o1, o2, action, dt, 2.0)
        if self.skill == 19:
            # back stand at 90 degrees
            return self.stand(o1, o2, action, dt, -1.6)
        if self.skill == 20:
            # back stand at 60 degrees.
            return self.stand(o1, o2, action, dt, -1.0)
        # if self.skill == 21:
        #     return self.stand(o1, o2, action, dt, -2.0)
        # Jumps
        if self.skill == 21:
            return self.jump(o1, o2, action, dt, 1.6)
        if self.skill == 22:
            return self.jump(o1, o2, action, dt, 1.0)
        if self.skill == 23:
            return self.jump(o1, o2, action, dt, -1.6)
        if self.skill == 24:
            return self.jump(o1, o2, action, dt, -1.0)

    def get_reward_vec(self, o1, o2, action, dt, num_skills):
        # assert self.skill is not None # batch reward doesn't require skill
        # Returns a reward vector corresponding to all rewards from all functions
        # storing original skill in temp var
        original_skill = self.skill
        reward_vec = np.zeros(num_skills)
        for i in range(num_skills):
            self.set_skill(i)
            reward_vec[i] = self.get_reward(o1, o2, action, dt)
        # Restoring original skill
        self.skill = original_skill
        return reward_vec

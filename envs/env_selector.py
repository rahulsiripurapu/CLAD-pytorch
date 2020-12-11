from envs.option_wrapper import optionwrap
from envs.gym_env import GymEnv


def option(env, config):
    if config.algo == "SAC":
        return env
    else:
        return optionwrap(env, config.num_skills)


def env_selector(config, seed=None):
    # if env_name == 'half-cheetah':
    #     env = HalfCheetahEnv()
    #     #TODO halfcheetah env doesn't have seed.. consider using gym env render?
    #     # But is deterministic anyways? WHY? range of eval rewards seems too low in comparison.
    #     # Is it perhaps because of the tanh in deterministic mode?
    #     return env
    if seed is None:
        seed = config.seed
    if config.env == 'half-cheetah':
        env = GymEnv('HalfCheetah-v1', seed=seed)
        return option(env, config)
    if config.env == 'hopper':
        env = GymEnv('Hopper-v1', seed=seed)
        return option(env, config)
    if config.env == 'double-pendulum':
        env = GymEnv('InvertedDoublePendulum-v1', seed=seed)
        return option(env, config)
    if config.env == 'humanoid':
        env = GymEnv('Humanoid-v1', seed=seed)
        return option(env, config)
    if config.env == 'ant':
        env = GymEnv('Ant-v1', seed=seed)
        return option(env, config)

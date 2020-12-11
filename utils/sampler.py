

class Sampler(object):

    def __init__(self, env, max_path_length):
        self.env = env
        self.max_path_length = max_path_length
        self.path_return = 0
        self.path_length = 0
        self.state = None
        self.reset()

    def reset(self, state=None):
        # TODO env can be reset from elsewhere! How to handle this: Use assert self.state == self.env.get_current_obs().
        self.path_length = 0
        self.path_return = 0
        self.state = self.env.reset(state)
        return self.state

    # def get_state(self):
    #     #TODO this function is wrong
    #     return self.env.get_current_obs()

    def sample(self, n_samples: int, policy=None):
        # TODO: is passing policy a good ideea here? instead use bool

        samples = []
        # TODO use get_state instead of self.state? What if env is changed elsewhere?
        # Env is being reset!
        # TODO assertion is horrible as it's a design flaw rrequiremnt
        assert (self.state == self.env.state).all()
        # print("Warning: Sampler env is being reset manually, because it doesn't accept skill as kwarg")
        # self.state = self.reset()
        for t in range(n_samples):
            if policy:
                action = policy.get_actions(self.state.reshape((1, -1)))
            else:
                action = self.env.action_space.sample()

            next_state, reward, done, info = self.env.step(action)
            # self.env.render(mode='human')
            self.path_return += reward
            self.path_length += 1
            # TODO: does not support parallel envs
            samples.append(dict(obs=self.state, action=action, next_obs=next_state, reward=reward, done=done,
                                info=info, path_length=self.path_length, path_return=self.path_return))

            if done or self.path_length >= self.max_path_length:
                self.reset()
            else:
                self.state = next_state

        return samples

# TODO: is it better to organize separately?
# def _get_empty_path_dict():
#     return dict(observations=[], actions=[], next_observations=[], rewards=[], dones=[], infos=[], agent_infos=[])

import os

class Config(object):

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    def as_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    @classmethod
    def set_seed(self, seed):
        if seed == 0:  # auto seed
            seed = int.from_bytes(os.urandom(3), 'little') + 1  # seed 0 is for `urandom`
        #TODO: generate log with seed
        #TODO: use lightning's seed_everything() instead of this set_seed method
        # (BUT DOES IT SEED GYM ENVS?)

        import numpy as np
        import torch
        import random
        np.random.seed(seed)
        torch.manual_seed(np.random.randint(2 ** 30))
        random.seed(np.random.randint(2 ** 30))
        torch.cuda.manual_seed_all(np.random.randint(2 ** 30))
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.benchmark = False
        #TODO: Set logic to set False when speed is important.
        # also set benchmark to True!
        torch.backends.cudnn.deterministic = True


    # TODO: test faking using argparse
# from argparse import Namespace
#
# args = {
#     'batch_size': 32,
#     'lr': 0.0002,
#     'b1': 0.5,
#     'b2': 0.999,
#     'latent_dim': 100
# }
# hparams = Namespace(**args)
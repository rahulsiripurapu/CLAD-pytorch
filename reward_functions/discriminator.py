import torch.nn as nn

from utils.mlp import MLP


class Discriminator(nn.Module):
    def __init__(self, input_dims, hidden_sizes, output_dims):
        #TODO: verify that input_dims, hidden sizes is a list
        super().__init__()
        self.mlp = MLP([input_dims, *hidden_sizes, output_dims], output_activation=None, squeeze=True)

    def forward(self, *inputs):
        # TODO: ensure inputs is a list of tensors
        return self.mlp(*inputs)
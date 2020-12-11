import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, blocks, activation=nn.ReLU, output_activation=nn.Tanh, squeeze=False,
                 w_initializer=nn.init.xavier_uniform_, b_initializer=nn.init.zeros_):
        super().__init__()

        self._blocks = blocks
        layers = []
        for in_features, out_features in zip(blocks[:-1], blocks[1:]):
            if layers and activation:
                layers.append(activation())
            layer = nn.Linear(in_features, out_features)
            w_initializer(layer.weight.data)
            b_initializer(layer.bias.data)
            layers.append(layer)
        if output_activation:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

        self._squeeze = squeeze
        self._activation = activation
        self._output_activation = output_activation

    def forward(self, *inputs):
        # Expects inputs to be a list of tensors with matching set of leading dimensions
        # otherwise cat would not be possible. TODO: support broadcasting
        if len(inputs) > 1:
            inputs = torch.cat(inputs, dim=-1)
        else:
            inputs = inputs[0]
        if self._squeeze:
            # TODO: verify dim for squeeze
            return torch.squeeze(self.net(inputs),dim=-1)
        else:
            return self.net(inputs)

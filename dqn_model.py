import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.input_is_image = len(input_shape) > 1

        if self.input_is_image:
            # shape should be [channels, height, width]
            conv_layers = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )

            conv_output_size = self._get_conv_output(input_shape, conv_layers)

            linear_layers = nn.Sequential(
                nn.Linear(conv_output_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )

            self.sequential_layers = nn.Sequential(
                conv_layers,
                nn.Flatten(),
                linear_layers,
            )
        else:
            self.sequential_layers = nn.Sequential(
                nn.Linear(input_shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
            )

    def _get_conv_output(self, shape, conv_layers):
        input = torch.rand(1, *shape)
        with torch.no_grad():
            output = conv_layers(input)
        return output.numel()

    def forward(self, x):
        return self.sequential_layers(x)

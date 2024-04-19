import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ImageDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ImageDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            self.fc1 = nn.Linear(self._get_conv_output(input_shape), 512)

        self.head = nn.Linear(512, n_actions)

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)  # shape should be [channels, height, width]
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.flatten(output)
        return output.shape[1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.head(x)

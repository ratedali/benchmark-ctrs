# This file is based on code publicly available at
# https://github.com/alinlab/smoothing-catrs
#
# Github Permalink: https://github.com/alinlab/smoothing-catrs/blob/d4bc576e7d373d158f087ba5744af8bb48466bb7/code/archs/lenet.py


import lightning as L
import torch.nn.functional as F
from torch import Tensor, nn


class LeNet(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

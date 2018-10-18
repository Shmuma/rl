import numpy as np

import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_shape, actions_count):
        super(Net, self).__init__()

        self.input_size = np.prod(input_shape)
        self.body = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.ELU(),
            nn.Linear(4096, 2048),
            nn.ELU()
        )
        self.policy = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512, actions_count)
        )
        self.value = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

    def forward(self, batch):
        x = batch.view((-1, self.input_size))
        body_out = self.body(x)
        return self.policy(body_out), self.value(body_out)

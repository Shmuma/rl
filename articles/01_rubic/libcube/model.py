import numpy as np

import torch.nn as nn

from . import cubes


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

    def forward(self, batch, value_only=False):
        x = batch.view((-1, self.input_size))
        body_out = self.body(x)
        value_out = self.value(body_out)
        if value_only:
            return value_out
        policy_out = self.policy(body_out)
        return policy_out, value_out


def encode_states(cube_env, states):
    assert isinstance(cube_env, cubes.CubeEnv)
    assert isinstance(states, (list, tuple))

    # states could be list of lists or just list of states
    if isinstance(states[0], list):
        encoded = np.zeros((len(states), len(states[0])) + cube_env.encoded_shape, dtype=np.float32)

        for i, st_list in enumerate(states):
            for j, state in enumerate(st_list):
                cube_env.encode_func(encoded[i, j], state)
    else:
        encoded = np.zeros((len(states), ) + cube_env.encoded_shape, dtype=np.float32)
        for i, state in enumerate(states):
            cube_env.encode_func(encoded[i], state)

    return encoded

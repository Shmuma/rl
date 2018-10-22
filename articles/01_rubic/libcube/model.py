import random
import numpy as np

import torch
import torch.nn as nn

from . import cubes


class Net(nn.Module):
    def __init__(self, input_shape, actions_count):
        super(Net, self).__init__()

        self.input_size = int(np.prod(input_shape))
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
                cube_env.encode_inplace(encoded[i, j], state)
    else:
        encoded = np.zeros((len(states), ) + cube_env.encoded_shape, dtype=np.float32)
        for i, state in enumerate(states):
            cube_env.encode_inplace(encoded[i], state)

    return encoded


def make_train_data(cube_env, net, device, batch_size, scramble_depth, shuffle=True):
    assert isinstance(cube_env, cubes.CubeEnv)
    # scramble cube states and their depths
    data = []
    rounds = batch_size // scramble_depth
    for _ in range(rounds):
        data.extend(cube_env.scramble_cube(scramble_depth, include_initial=False))
    if shuffle:
        random.shuffle(data)
    cube_depths, cube_states = zip(*data)

    # explore each state by doing 1-step BFS search and keep a mask of goal states (for reward calculation)
    explored_states, explored_goals = [], []
#    goal_indices = []
    for idx, s in enumerate(cube_states):
        states, goals = cube_env.explore_state(s)
        explored_states.append(states)
        explored_goals.append(goals)
#        if cube_env.is_goal(s):
#            goal_indices.append(idx)

    # obtain network's values for all explored states
    enc_explored = encode_states(cube_env, explored_states)           # shape: (states, actions, encoded_shape)
    shape = enc_explored.shape
    enc_explored_t = torch.tensor(enc_explored).to(device)
    enc_explored_t = enc_explored_t.view(shape[0]*shape[1], *shape[2:])     # shape: (states*actions, encoded_shape)
    value_t = net(enc_explored_t, value_only=True)
    value_t = value_t.squeeze(-1).view(shape[0], shape[1])                  # shape: (states, actions)
    # add reward to the values
    goals_mask_t = torch.tensor(explored_goals, dtype=torch.int8).to(device)
    goals_mask_t += goals_mask_t - 1                                        # has 1 at final states and -1 elsewhere
    value_t += goals_mask_t.type(dtype=torch.float32)

    # find target value and target policy
    max_val_t, max_act_t = value_t.max(dim=1)
#    max_val_t[goal_indices] = 1.0
#    max_act_t[goal_indices] = 0

    # create train input
    enc_input = encode_states(cube_env, cube_states)
    enc_input_t = torch.tensor(enc_input).to(device)
    cube_depths_t = torch.tensor(cube_depths, dtype=torch.float32).to(device)
    weights_t = 1/cube_depths_t
    return enc_input_t.detach(), weights_t.detach(), max_act_t.detach(), max_val_t.detach()

#!/usr/bin/env python3
"""
Ad-hoc utility to analyze trained model and various training process details
"""
import argparse
import logging

import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from libcube import cubes
from libcube import model

log = logging.getLogger("train_debug")


# How many data to generate for plots
MAX_DEPTH = 10
ROUND_COUNTS = 100
# debug params
#MAX_DEPTH = 5
#ROUND_COUNTS = 2


def gen_states(cube_env, max_depth, round_counts):
    """
    Generate random states of various scramble depth
    :param cube_env: CubeEnv instance
    :return: list of list of (state, correct_action_index) pairs
    """
    assert isinstance(cube_env, cubes.CubeEnv)

    result = [[] for _ in range(max_depth)]
    for _ in range(round_counts):
        data = cube_env.scramble_cube(max_depth, return_inverse=True)
        for depth, state, inv_action in data:
            result[depth-1].append((state, inv_action.value))
    return result


def make_train_data(cube_env, net, device="cpu", scramble_depth=2):
    data = []
    rounds = 10 // scramble_depth
    for _ in range(rounds):
        data.extend(cube_env.scramble_cube(scramble_depth))
    cube_depths, cube_states = zip(*data)

    # explore each state by doing 1-step BFS search and keep a mask of goal states (for reward calculation)
    explored_states, explored_goals = [], []
    for s in cube_states:
        states, goals = cube_env.explore_state(s)
        explored_states.append(states)
        explored_goals.append(goals)

    # obtain network's values for all explored states
    enc_explored = model.encode_states(cube_env, explored_states)           # shape: (states, actions, encoded_shape)
    shape = enc_explored.shape
    enc_explored_t = torch.tensor(enc_explored).to(device)
    enc_explored_t = enc_explored_t.view(shape[0]*shape[1], *shape[2:])     # shape: (states*actions, encoded_shape)
    policy_t, value_t = net(enc_explored_t)
    value_t = value_t.squeeze(-1).view(shape[0], shape[1])                  # shape: (states, actions)
    # add reward to the values
    goals_mask_t = torch.tensor(explored_goals, dtype=torch.int8).to(device)
    goals_mask_t += goals_mask_t - 1                                        # has 1 at final states and -1 elsewhere
    value_t = value_t.clamp(-1, 1)
    value_t = torch.max(value_t, goals_mask_t.type(dtype=torch.float32))
    # find target value and target policy
    max_val_t, max_act_t = value_t.max(dim=1)

    # create train input
    enc_input = model.encode_states(cube_env, cube_states)
    enc_input_t = torch.tensor(enc_input).to(device)
    cube_depths_t = torch.tensor(cube_depths, dtype=torch.float32).to(device)
    weights_t = 1/cube_depths_t
    return enc_input_t, weights_t, max_act_t, max_val_t


if __name__ == "__main__":
    sns.set()

    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", required=True, help="Type of env to train, supported types=%s" % cubes.names())
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-o", "--output", required=True, help="Output prefix for plots")
    args = parser.parse_args()

    cube_env = cubes.get(args.env)
    log.info("Selected cube: %s", cube_env)
    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum))
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    net.eval()
    log.info("Network loaded from %s", args.model)

#    make_train_data(cube_env, net)

    states_by_depth = gen_states(cube_env, max_depth=MAX_DEPTH, round_counts=ROUND_COUNTS)
    # for idx, states in enumerate(states_by_depth):
    #     log.info("%d: %s", idx, states)

    # flatten returned data
    data = []
    for depth, states in enumerate(states_by_depth):
        for s, inv_action in states:
            data.append((depth+1, s, inv_action))
    depths, states, inv_actions = map(list, zip(*data))

    # process states with net
    enc_states = model.encode_states(cube_env, states)
    enc_states_t = torch.tensor(enc_states)
    policy_t, value_t = net(enc_states_t)
    value_t = value_t.squeeze(-1)
    value = value_t.cpu().detach().numpy()
    policy = F.softmax(policy_t, dim=1).cpu().detach().numpy()

    # plot value per depth of scramble
    plot = sns.lineplot(depths, value)
    plot.set_title("Values per depths")
    plot.get_figure().savefig(args.output + "-vals_vs_depths.png")

    # plot action match
    plt.clf()
    actions = np.argmax(policy, axis=1)
    actions_match = (actions == inv_actions).astype(np.int8)
    plot = sns.lineplot(depths, actions_match)
    plot.set_title("Actions accuracy per depths")
    plot.get_figure().savefig(args.output + "-acts_vs_depths.png")

    pass

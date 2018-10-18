#!/usr/bin/env python3
import argparse
import logging
import numpy as np

from libcube import cubes
from libcube import model

log = logging.getLogger("train")

SCRAMBLES_COUNT = 10
ROUNDS_COUNT = 2


def make_train_data(cube_env):
    # scramble cube states and their depths
    data = []
    for _ in range(ROUNDS_COUNT):
        data.extend(cubes.scramble_cube(cube_env, SCRAMBLES_COUNT))
    cube_depths, cube_states = zip(*data)
    # explore each state by doing 1-step BFS search
    explored_states = [cubes.explore_state(cube_env, s) for s in cube_states]
    # obtain network's values for all explored states
    enc_input = model.encode_states(cube_env, explored_states)
    print(enc_input.shape)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cube", required=True, help="Type of cube to train, supported types=%s" % cubes.names())
    args = parser.parse_args()

    cube_env = cubes.get(args.cube)
    assert isinstance(cube_env, cubes.CubeEnv)

    log.info("Selected cube: %s", cube_env)

    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum))
    print(net)

    make_train_data(cube_env)

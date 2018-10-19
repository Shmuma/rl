#!/usr/bin/env python3
"""
Solver using MCTS and trained model
"""
import argparse
import logging

import torch

from libcube import cubes
from libcube import model



log = logging.getLogger("solver")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", required=True, help="Type of env to train, supported types=%s" % cubes.names())
    parser.add_argument("-m", "--model", required=True, help="Model file to load, has to match env type")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Text file with permutations to read cubes to solve, "
                                             "possibly produced by gen_cubes.py")
    group.add_argument("-p", "--perms", help="Permutation in form of actions list separated by spaces")
    group.add_argument("-r", "--random", metavar="DEPTH", type=int, help="Generate random scramble of given depth")
    args = parser.parse_args()

    cube_env = cubes.get(args.env)
    assert isinstance(cube_env, cubes.CubeEnv)              # just to help pycharm understand type

    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum))
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    print(net)
    pass



#!/usr/bin/env python3
"""
Solver using MCTS and trained model
"""
import argparse
import logging
import random

import torch

from libcube import cubes
from libcube import model

log = logging.getLogger("solver")


def generate_task(env, depth):
    return [env.sample_action().value for _ in range(depth)]


def solve_task(env, task, net, cube_idx=None):
    log_prefix = "" if cube_idx is None else "cube %d: " % cube_idx
    log.info("%sGot task %s, solving...", log_prefix, task)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", required=True, help="Type of env to train, supported types=%s" % cubes.names())
    parser.add_argument("-m", "--model", required=True, help="Model file to load, has to match env type")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Text file with permutations to read cubes to solve, "
                                             "possibly produced by gen_cubes.py")
    group.add_argument("-p", "--perm", help="Permutation in form of actions list separated by comma")
    group.add_argument("-r", "--random", metavar="DEPTH", type=int, help="Generate random scramble of given depth")
    args = parser.parse_args()

    cube_env = cubes.get(args.env)
    log.info("Using environment %s", cube_env)
    assert isinstance(cube_env, cubes.CubeEnv)              # just to help pycharm understand type

    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum))
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    net.eval()
    log.info("Network loaded from %s", args.model)

    if args.random is not None:
        task = generate_task(cube_env, args.random)
        solve_task(cube_env, task, net)
    elif args.perm is not None:
        task = list(map(int, args.perm.split(',')))
        solve_task(cube_env, task, net)
    elif args.input is not None:
        with open(args.input, 'rt', encoding='utf-8') as fd:
            for idx, l in enumerate(fd):
                task = list(map(int, l.strip().split(',')))
                solve_task(cube_env, task, net, cube_idx=idx)


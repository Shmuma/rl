#!/usr/bin/env python3
"""
Solver using MCTS and trained model
"""
import time
import argparse
import logging
import itertools

import torch

from libcube import cubes
from libcube import model
from libcube import mcts

log = logging.getLogger("solver")


DEFAULT_MAX_SECONDS = 60


def generate_task(env, depth):
    return [env.sample_action().value for _ in range(depth)]


def solve_task(env, task, net, cube_idx=None, max_seconds=DEFAULT_MAX_SECONDS, device="cpu"):
    log_prefix = "" if cube_idx is None else "cube %d: " % cube_idx
    log.info("%sGot task %s, solving...", log_prefix, task)
    cube_state = env.scramble(map(env.action_enum, task))
    tree = mcts.MCTS(env, cube_state, device=device)
    step_no = 0
    ts = time.time()

    while True:
        r = tree.search(net)
        if r:
            log.info("On step %d we found goal state, unroll. Speed %.2f searches/s",
                     step_no, step_no / (time.time() - ts))
            break
        step_no += 1
        if time.time() - ts > max_seconds:
            log.info("Time is up, cube wasn't solved. Did %d searches, speed %.2f searches/s..",
                     step_no, step_no / (time.time() - ts))
            for _, cnt in zip(range(10), tree.act_counts.values()):
                print(cnt)
            for _, cnt in zip(range(10), tree.val_maxes.values()):
                print(cnt)
            break


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", required=True, help="Type of env to train, supported types=%s" % cubes.names())
    parser.add_argument("-m", "--model", required=True, help="Model file to load, has to match env type")
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_SECONDS,
                        help="Limit in seconds for each task, default=%s" % DEFAULT_MAX_SECONDS)
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Text file with permutations to read cubes to solve, "
                                             "possibly produced by gen_cubes.py")
    group.add_argument("-p", "--perm", help="Permutation in form of actions list separated by comma")
    group.add_argument("-r", "--random", metavar="DEPTH", type=int, help="Generate random scramble of given depth")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    cube_env = cubes.get(args.env)
    log.info("Using environment %s", cube_env)
    assert isinstance(cube_env, cubes.CubeEnv)              # just to help pycharm understand type

    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum)).to(device)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    net.eval()
    log.info("Network loaded from %s", args.model)

    if args.random is not None:
        task = generate_task(cube_env, args.random)
        solve_task(cube_env, task, net, max_seconds=args.max_time, device=device)
    elif args.perm is not None:
        task = list(map(int, args.perm.split(',')))
        solve_task(cube_env, task, net, max_seconds=args.max_time, device=device)
    elif args.input is not None:
        log.info("Processing scrambles from %s", args.input)
        with open(args.input, 'rt', encoding='utf-8') as fd:
            for idx, l in enumerate(fd):
                task = list(map(int, l.strip().split(',')))
                solve_task(cube_env, task, net, cube_idx=idx, max_seconds=args.max_time, device=device)


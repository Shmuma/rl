#!/usr/bin/env python3
import os
import time
import argparse
import logging
import random
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from libcube import cubes
from libcube import model

log = logging.getLogger("train")

CUBES_PER_BATCH = 1000
DEFAULT_SCRAMBLE_DEPTH = 200
REPORT_ITERS = 100
CHECKPOINT_ITERS = 1000
LEARNING_RATE = 5e-5
LR_DECAY_ITERS = 5000


def make_train_data(cube_env, net, device, use_rqsrt=False, scramble_depth=DEFAULT_SCRAMBLE_DEPTH):
    net.eval()
    # scramble cube states and their depths
    data = []
    rounds = CUBES_PER_BATCH // scramble_depth
    for _ in range(rounds):
        data.extend(cubes.scramble_cube(cube_env, scramble_depth))
    random.shuffle(data)
    cube_depths, cube_states = zip(*data)

    # explore each state by doing 1-step BFS search and keep a mask of goal states (for reward calculation)
    explored_states, explored_goals = [], []
    for s in cube_states:
        states, goals = cubes.explore_state(cube_env, s)
        explored_states.append(states)
        explored_goals.append(goals)

    # obtain network's values for all explored states
    enc_explored = model.encode_states(cube_env, explored_states)           # shape: (states, actions, encoded_shape)
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

    # create train input
    enc_input = model.encode_states(cube_env, cube_states)
    enc_input_t = torch.tensor(enc_input).to(device)
    cube_depths_t = torch.tensor(cube_depths, dtype=torch.float32).to(device)
    if use_rqsrt:
        weights_t = torch.rsqrt(cube_depths_t)
    else:
        weights_t = 1/cube_depths_t
    net.train()
    return enc_input_t, weights_t, max_act_t, max_val_t


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", required=True, help="Type of env to train, supported types=%s" % cubes.names())
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", action="store_true", help="Enable cuda")
    parser.add_argument("--rsqrt", action="store_true", default=False, help="Use 1/sqrt(D) weight instead of 1/D")
    parser.add_argument("--depth", type=int, default=DEFAULT_SCRAMBLE_DEPTH,
                        help="Depth of cube scramble, default=%s" % DEFAULT_SCRAMBLE_DEPTH)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    name = "%s-d%d-%s" % (args.env, args.depth, args.name)
    writer = SummaryWriter(comment="-" + name)
    save_path = os.path.join("saves", name)
    os.makedirs(save_path)

    cube_env = cubes.get(args.env)
    assert isinstance(cube_env, cubes.CubeEnv)

    log.info("Selected cube: %s", cube_env)

    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum)).to(device)
    print(net)
    opt = optim.RMSprop(net.parameters(), lr=LEARNING_RATE)
    sched = scheduler.LambdaLR(opt, lr_lambda=lambda e: e*0.95)

    step_idx = 0
    buf_policy_loss, buf_value_loss, buf_loss = [], [], []
    buf_policy_loss_raw, buf_value_loss_raw, buf_loss_raw = [], [], []
    ts = time.time()
    best_loss = None

    while True:
        if step_idx % LR_DECAY_ITERS == 0:
            sched.step()
            writer.add_scalar("lr", sched.get_lr()[0], step_idx)

        step_idx += 1
        x_t, weights_t, y_policy_t, y_value_t = make_train_data(cube_env, net, device, use_rqsrt=args.rsqrt,
                                                                scramble_depth=args.depth)
        opt.zero_grad()
        policy_out_t, value_out_t = net(x_t)
        value_out_t = value_out_t.squeeze(-1)
        value_loss_t = (value_out_t - y_value_t)**2
        value_loss_raw_t = value_loss_t.mean()
        value_loss_t *= weights_t
        value_loss_t = value_loss_t.mean()
        policy_loss_t = F.cross_entropy(policy_out_t, y_policy_t, reduction='none')
        policy_loss_raw_t = policy_loss_t.mean()
        policy_loss_t *= weights_t
        policy_loss_t = policy_loss_t.mean()
        loss_raw_t = policy_loss_raw_t + value_loss_raw_t
        loss_t = value_loss_t + policy_loss_t
        loss_t.backward()
        opt.step()

        # save losses
        buf_policy_loss.append(policy_loss_t.item())
        buf_value_loss.append(value_loss_t.item())
        buf_loss.append(loss_t.item())
        buf_loss_raw.append(loss_raw_t.item())
        buf_value_loss_raw.append(value_loss_raw_t.item())
        buf_policy_loss_raw.append(policy_loss_raw_t.item())

        if step_idx % REPORT_ITERS == 0:
            m_policy_loss = np.mean(buf_policy_loss)
            m_value_loss = np.mean(buf_value_loss)
            m_loss = np.mean(buf_loss)
            buf_value_loss.clear()
            buf_policy_loss.clear()
            buf_loss.clear()
            m_policy_loss_raw = np.mean(buf_policy_loss_raw)
            m_value_loss_raw = np.mean(buf_value_loss_raw)
            m_loss_raw = np.mean(buf_loss_raw)
            buf_value_loss_raw.clear()
            buf_policy_loss_raw.clear()
            buf_loss_raw.clear()

            dt = time.time() - ts
            ts = time.time()
            speed = CUBES_PER_BATCH * REPORT_ITERS / dt
            log.info("%d: p_loss=%.3e, v_loss=%.3e, loss=%.3e, speed=%.1f cubes/s",
                     step_idx, m_policy_loss, m_value_loss, m_loss, speed)
            writer.add_scalar("loss_policy", m_policy_loss, step_idx)
            writer.add_scalar("loss_value", m_value_loss, step_idx)
            writer.add_scalar("loss", m_loss, step_idx)
            writer.add_scalar("loss_policy_raw", m_policy_loss_raw, step_idx)
            writer.add_scalar("loss_value_raw", m_value_loss_raw, step_idx)
            writer.add_scalar("loss_raw", m_loss_raw, step_idx)
            writer.add_scalar("speed", speed, step_idx)

            if best_loss is None:
                best_loss = m_loss
            elif best_loss > m_loss:
                name = os.path.join(save_path, "best_%.4e.dat" % m_loss)
                torch.save(net.state_dict(), name)
                best_loss = m_loss

        if step_idx % CHECKPOINT_ITERS == 0:
            name = os.path.join(save_path, "chpt_%06d.dat" % step_idx)
            torch.save(net.state_dict(), name)


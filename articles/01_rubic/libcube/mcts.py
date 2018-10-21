import random
import numpy as np
import collections

from . import cubes
from . import model

import torch
import torch.nn.functional as F


class MCTS:
    """
    Monte Carlo Tree Search state and method
    """
    def __init__(self, cube_env, state, exploration_c=10000, virt_loss_nu=10.0, device="cpu", random_action_depth=0):
        assert isinstance(cube_env, cubes.CubeEnv)
        assert cube_env.is_state(state)

        self.cube_env = cube_env
        self.root_state = state
        self.exploration_c = exploration_c
        self.virt_loss_nu = virt_loss_nu
        self.random_action_depth = random_action_depth
        self.device = device

        # Tree state
        shape = (len(cube_env.action_enum), )
        # correspond to N_s(a) in the paper
        self.act_counts = collections.defaultdict(lambda: np.zeros(shape, dtype=np.uint32))
        # correspond to W_s(a)
        self.val_maxes = collections.defaultdict(lambda: np.zeros(shape, dtype=np.float32))
        # correspond to P_s(a)
        self.prob_actions = {}
        # correspond to L_s(a)
        self.virt_loss = collections.defaultdict(lambda: np.zeros(shape, dtype=np.float32))
        # TODO: check speed and memory of edge-less version
        self.edges = {}

    def __repr__(self):
        return "MCTS(states=%d)" % len(self.edges)

    def dump_root(self):
        print("Root state:")
        self.dump_state(self.root_state)
        # states, _ = cubes.explore_state(self.cube_env, self.root_state)
        # for idx, s in enumerate(states):
        #     print("")
        #     print("State %d" % idx)
        #     self.dump_state(s)

    def dump_state(self, s):
        print("")
        print("act_counts: %s" % ", ".join(map(lambda v: "%8d" % v, self.act_counts[s].tolist())))
        print("probs:      %s" % ", ".join(map(lambda v: "%.2e" % v, self.prob_actions[s].tolist())))
        print("val_maxes:  %s" % ", ".join(map(lambda v: "%.2e" % v, self.val_maxes[s].tolist())))

        act_counts = self.act_counts[s]
        N_sqrt = np.sqrt(np.sum(act_counts))
        u = self.exploration_c * N_sqrt / (act_counts + 1)
        print("u:          %s" % ", ".join(map(lambda v: "%.2e" % v, u.tolist())))
        u *= self.prob_actions[s]
        print("u*prob:     %s" % ", ".join(map(lambda v: "%.2e" % v, u.tolist())))
        q = self.val_maxes[s] - self.virt_loss[s]
        print("q:          %s" % ", ".join(map(lambda v: "%.2e" % v, q.tolist())))
        fin = u + q
        print("u*prob + q: %s" % ", ".join(map(lambda v: "%.2e" % v, fin.tolist())))
        act = np.argmax(fin, axis=0)
        print("Action: %s" % act)

    def search(self, net):
        s = self.root_state
        path_actions = []
        path_states = []

        # walking down the tree
        d = 0
        while True:
            next_states = self.edges.get(s)
            if next_states is None:
                break

            act_counts = self.act_counts[s]
            N_sqrt = np.sqrt(np.sum(act_counts))
            if d < self.random_action_depth or N_sqrt < 1e-6:
                act = random.randrange(len(self.cube_env.action_enum))
            else:
                u = self.exploration_c * N_sqrt / (act_counts + 1)
                u *= self.prob_actions[s]
                q = self.val_maxes[s] - self.virt_loss[s]
                act = np.argmax(u + q)
            self.virt_loss[s][act] += self.virt_loss_nu
            path_actions.append(act)
            path_states.append(s)
            s = next_states[act]
            d += 1

        # reached the leaf state, expand it
        child_states, child_goal = self.cube_env.explore_state(s)
        self.edges[s] = child_states

        # calculate policy and values for our states
        eval_policy, eval_values = self.evaluate_states(net, [s] + child_states, self.device)

        # we can miss policy for start state, save it
        if s not in self.prob_actions:
            self.prob_actions[s] = eval_policy[0]
        # save policy output for children
        for child_s, policy in zip(child_states, eval_policy[1:]):
            self.prob_actions[child_s] = policy
        # value of expanded state to be backed up
        value = eval_values[0]

        # back up our path
        for path_s, path_a in zip(path_states, path_actions):
            self.act_counts[path_s][path_a] += 1
            w = self.val_maxes[path_s]
            w[path_a] = max(w[path_a], value)
            self.virt_loss[path_s][path_a] -= self.virt_loss_nu

        return np.any(child_goal)

    def evaluate_states(self, net, states, device):
        """
        Ask network to return policy and values
        :param net:
        :param states:
        :return:
        """
        enc_states = model.encode_states(self.cube_env, states)
        enc_states_t = torch.tensor(enc_states).to(device)
        policy_t, value_t = net(enc_states_t)
        policy_t = F.softmax(policy_t, dim=1)
        return policy_t.detach().cpu().numpy(), value_t.squeeze(-1).detach().cpu().numpy()


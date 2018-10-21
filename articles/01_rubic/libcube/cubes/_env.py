"""
Generic cube env representation and registry
"""
import logging
import random

log = logging.getLogger("cube.env")
_registry = {}


class CubeEnv:
    def __init__(self, name, state_type, initial_state, is_goal_pred,
                 action_enum, transform_func, render_func, encoded_shape,
                 encode_func):
        self.name = name
        self._state_type = state_type
        self.initial_state = initial_state
        self._is_goal_pred = is_goal_pred
        self.action_enum = action_enum
        self._transform_func = transform_func
        self._render_func = render_func
        self.encoded_shape = encoded_shape
        self._encode_func = encode_func

    def __repr__(self):
        return "CubeEnv(%r)" % self.name

    # wrapper functions
    def is_goal(self, state):
        assert isinstance(state, self._state_type)
        return self._is_goal_pred(state)

    def transform(self, state, action):
        assert isinstance(state, self._state_type)
        assert isinstance(action, self.action_enum)
        return self._transform_func(state, action)

    def render(self, state):
        assert isinstance(state, self._state_type)
        return self._render_func(state)

    def encode_inplace(self, target, state):
        assert isinstance(state, self._state_type)
        return self._encode_func(target, state)

    # Utility functions
    def sample_action(self):
        return self.action_enum(random.randrange(len(self.action_enum)))

    def scramble(self, actions):
        s = self.initial_state
        for action in actions:
            s = self.transform(s, action)
        return s

    def is_state(self, state):
        return isinstance(state, self._state_type)

    def scramble_cube(self, scrambles_count):
        """
        Generate sequence of random cube scrambles
        :param scrambles_count: count of scrambles to perform
        :return: list of tuples (depth, state)
        """
        assert isinstance(scrambles_count, int)
        assert scrambles_count > 0

        state = self.initial_state
        result = []
        for depth in range(scrambles_count):
            state = self.transform(state, self.sample_action())
            result.append((depth+1, state))
        return result

    def explore_state(self, state):
        """
        Expand cube state by applying every action to it
        :param state: state to explore
        :return: tuple of two lists: [states reachable], [flag that state is initial]
        """
        res_states, res_flags = [], []
        for action in self.action_enum:
            new_state = self.transform(state, action)
            is_init = self.is_goal(new_state)
            res_states.append(new_state)
            res_flags.append(is_init)
        return res_states, res_flags


def register(cube_env):
    assert isinstance(cube_env, CubeEnv)
    global _registry

    if cube_env.name in _registry:
        log.warning("Cube environment %s is already registered, ignored", cube_env)
    else:
        _registry[cube_env.name] = cube_env


def get(name):
    assert isinstance(name, str)
    return _registry.get(name)


def names():
    return list(sorted(_registry.keys()))

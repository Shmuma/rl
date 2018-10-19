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
        self.state_type = state_type
        self.initial_state = initial_state
        self.is_goal_pred = is_goal_pred
        self.action_enum = action_enum
        self.transform_func = transform_func
        self.render_func = render_func
        self.encoded_shape = encoded_shape
        self.encode_func = encode_func

    def __repr__(self):
        return "CubeEnv(%r)" % self.name

    def sample_action(self):
        return self.action_enum(random.randrange(len(self.action_enum)))


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

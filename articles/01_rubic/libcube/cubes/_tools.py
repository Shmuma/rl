"""
Utility functions generic for all cube types
"""
from . import _env


def scramble_cube(cube_env, scrambles_count):
    """
    Generate sequence of random cube scrambles
    :param cube_env: CubeEnv instance
    :param scrambles_count: count of scrambles to perform
    :return: list of tuples (depth, state)
    """
    assert isinstance(cube_env, _env.CubeEnv)
    assert isinstance(scrambles_count, int)
    assert scrambles_count > 0

    state = cube_env.initial_state
    result = []
    for depth in range(scrambles_count):
        state = cube_env.transform_func(state, cube_env.sample_action())
        result.append((depth+1, state))
    return result


def explore_state(cube_env, state):
    """
    Expand cube state by applying every action to it
    :param cube_env: CubeEnv instance
    :param state: state to explore
    :return: list of states reachable from the state
    """
    assert isinstance(cube_env, _env.CubeEnv)
    assert isinstance(state, cube_env.state_type)
    return [cube_env.transform_func(state, action) for action in cube_env.action_enum]
"""
Classic cube 3x3
"""
import collections

# environment API
State = collections.namedtuple("State", field_names=['corner_pos', 'side_pos', 'corner_ort', 'side_ort'])

# rendered state -- list of colors of every side
RenderedState = collections.namedtuple("RenderedState", field_names=['top', 'front', 'left', 'right', 'back', 'bottom'])

# initial (solved state)
initial_state = State(corner_pos=range(8), side_pos=range(12), corner_ort=[0]*8, side_ort=[0]*12)

# available actions. Capital actions denote clockwise rotation
action_names = ['R', 'L', 'T', 'B', 'F', 'B', 'r', 'l', 't', 'b', 'f', 'b']
actions = list(range(len(action_names)))


# apply action to the state
def transform(state, action):
    assert isinstance(state, State)
    assert isinstance(action, int)


# make initial state of rendered side
def _init_side(color):
    return [color if idx == 4 else None for idx in range(9)]


# create initial sides in the right order
def _init_sides():
    return [
        _init_side('W'),    # top
        _init_side('G'),    # left
        _init_side('O'),    # back
        _init_side('R'),    # front
        _init_side('B'),    # right
        _init_side('Y')     # bottom
    ]


# orient corner cubelet
def _map_orient(cols, orient_id):
    if orient_id == 0:
        return cols
    elif orient_id == 1:
        return cols[2], cols[0], cols[1]
    else:
        return cols[1], cols[2], cols[0]


# corner cubelets colors (clockwise from main label). Order of cubelets are first top,
# in counter-clockwise, started from front left
corner_colors = (
    ('W', 'R', 'G'), ('W', 'B', 'R'), ('W', 'O', 'B'), ('W', 'G', 'O'),
    ('Y', 'G', 'R'), ('Y', 'R', 'B'), ('Y', 'B', 'O'), ('Y', 'O', 'G')
)

side_colors = (
    ('W', 'R'), ('W', 'B'), ('W', 'O'), ('W', 'G'),
    ('R', 'G'), ('R', 'B'), ('O', 'B'), ('O', 'G'),
    ('Y', 'R'), ('Y', 'B'), ('Y', 'O'), ('Y', 'G')
)


# map every 3-side cubelet to their projection on sides
# sides are indexed in the order of _init_sides() function result
corner_maps = (
    # top layer
    ((0, 6), (3, 0), (1, 2)),
    ((0, 8), (4, 0), (3, 2)),
    ((0, 2), (2, 0), (4, 2)),
    ((0, 0), (1, 0), (2, 2)),
    # bottom layer
    ((5, 0), (1, 8), (3, 6)),
    ((5, 2), (3, 8), (4, 6)),
    ((5, 8), (4, 8), (2, 6)),
    ((5, 6), (2, 8), (1, 6))
)

# map every 2-side cubelet to their projection on sides
side_maps = (
    # top layer
    ((0, 7), (3, 1)),
    ((0, 5), (4, 1)),
    ((0, 1), (2, 1)),
    ((0, 3), (1, 1)),
    # middle layer
    ((3, 3), (1, 5)),
    ((3, 5), (4, 3)),
    ((2, 3), (4, 5)),
    ((2, 5), (1, 3)),
    # bottom layer
    ((5, 1), (3, 7)),
    ((5, 5), (4, 7)),
    ((5, 7), (2, 7)),
    ((5, 3), (1, 7))
)


# render state into human readable form
def render(state):
    assert isinstance(state, State)
    global corner_colors, corner_maps, side_colors, side_maps

    sides = _init_sides()

    for corner, orient, maps in zip(state.corner_pos, state.corner_ort, corner_maps):
        cols = corner_colors[corner]
        cols = _map_orient(cols, orient)
        for (arr_idx, index), col in zip(maps, cols):
            sides[arr_idx][index] = col

    for side, orient, maps in zip(state.side_pos, state.side_ort, side_maps):
        cols = side_colors[side]
        cols = cols if orient == 0 else (cols[1], cols[0])
        for (arr_idx, index), col in zip(maps, cols):
            sides[arr_idx][index] = col

    return RenderedState(top=sides[0], left=sides[1], back=sides[2], front=sides[3],
                         right=sides[4], bottom=sides[5])
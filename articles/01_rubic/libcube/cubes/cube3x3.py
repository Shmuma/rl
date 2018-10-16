"""
Classic cube 3x3
"""
import enum
import collections

# environment API
State = collections.namedtuple("State", field_names=['corner_pos', 'side_pos', 'corner_ort', 'side_ort'])

# rendered state -- list of colors of every side
RenderedState = collections.namedtuple("RenderedState", field_names=['top', 'front', 'left', 'right', 'back', 'bottom'])

# initial (solved state)
initial_state = State(corner_pos=range(8), side_pos=range(12), corner_ort=[0]*8, side_ort=[0]*12)

# available actions. Capital actions denote clockwise rotation
class Action(enum.Enum):
    R = 0
    L = 1
    T = 2
    D = 3
    F = 4
    B = 5
    r = 6
    l = 7
    t = 8
    d = 9
    f = 10
    b = 11


def _permute(t, m):
    """
    Perform permutation of tuple according to mapping m
    """
    r = list(t)
    for from_idx, to_idx in m:
        r[to_idx] = t[from_idx]
    return r


def _rotate(corner_ort, corners):
    """
    Rotate given corners 120 degrees
    """
    r = list(corner_ort)
    for c in corners:
        r[c] = (r[c] + 1) % 3
    return r


def _flip(side_ort, sides):
    return [
        o if idx not in sides else 1-o
        for idx, o in enumerate(side_ort)
    ]


# apply action to the state
def transform(state, action):
    assert isinstance(state, State)
    assert isinstance(action, Action)

    if action == Action.R:
        m = ((1, 2), (2, 6), (6, 5), (5, 1))
        s = ((1, 6), (6, 9), (9, 5), (5, 1))
        c_o = (1, 1, 2, 5, 6, 6)
        return State(corner_pos=_permute(state.corner_pos, m),
                     corner_ort=_rotate(state.corner_ort, c_o),
                     side_pos=_permute(state.side_pos, s),
                     side_ort=state.side_ort)
    elif action == Action.r:
        m = ((2, 1), (6, 2), (5, 6), (1, 5))
        s = ((6, 1), (9, 6), (5, 9), (1, 5))
        c_o = (1, 1, 2, 5, 6, 6)
        return State(corner_pos=_permute(state.corner_pos, m),
                     corner_ort=_rotate(state.corner_ort, c_o),
                     side_pos=_permute(state.side_pos, s),
                     side_ort=state.side_ort)
    elif action == Action.L:
        m = ((3, 0), (7, 3), (0, 4), (4, 7))
        s = ((7, 3), (3, 4), (11, 7), (4, 11))
        c_o = (0, 3, 3, 4, 4, 7)
        return State(corner_pos=_permute(state.corner_pos, m),
                     corner_ort=_rotate(state.corner_ort, c_o),
                     side_pos=_permute(state.side_pos, s),
                     side_ort=state.side_ort)
    elif action == Action.l:
        m = ((0, 3), (3, 7), (4, 0), (7, 4))
        s = ((3, 7), (4, 3), (7, 11), (11, 4))
        c_o = (0, 3, 3, 4, 4, 7)
        return State(corner_pos=_permute(state.corner_pos, m),
                     corner_ort=_rotate(state.corner_ort, c_o),
                     side_pos=_permute(state.side_pos, s),
                     side_ort=state.side_ort)
    elif action == Action.F:
        pass
    elif action == Action.f:
        pass
    elif action == Action.B:
        pass
    elif action == Action.b:
        pass
    elif action == Action.T:
        m = ((0, 3), (1, 0), (2, 1), (3, 2))
        return State(corner_pos=_permute(state.corner_pos, m),
                     corner_ort=state.corner_ort,
                     side_pos=_permute(state.side_pos, m),
                     side_ort=state.side_ort)
    elif action == Action.t:
        m = ((0, 1), (1, 2), (2, 3), (3, 0))
        return State(corner_pos=_permute(state.corner_pos, m),
                     corner_ort=state.corner_ort,
                     side_pos=_permute(state.side_pos, m),
                     side_ort=state.side_ort)
    elif action == Action.D:
        m = ((4, 5), (5, 6), (6, 7), (7, 4))
        s = ((8, 9), (9, 10), (10, 11), (11, 8))
        return State(corner_pos=_permute(state.corner_pos, m),
                     corner_ort=state.corner_ort,
                     side_pos=_permute(state.side_pos, s),
                     side_ort=state.side_ort)
    elif action == Action.d:
        m = ((4, 7), (5, 4), (6, 5), (7, 6))
        s = ((8, 11), (9, 8), (10, 9), (11, 10))
        return State(corner_pos=_permute(state.corner_pos, m),
                     corner_ort=state.corner_ort,
                     side_pos=_permute(state.side_pos, s),
                     side_ort=state.side_ort)


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
import unittest

from libcube.cubes import cube3x3


class CubeRender(unittest.TestCase):
    def test_init_render(self):
        state = cube3x3.initial_state
        render = cube3x3.render(state)
        self.assertIsInstance(render, cube3x3.RenderedState)
        self.assertEqual(render.top, ['W'] * 9)
        self.assertEqual(render.back, ['O'] * 9)
        self.assertEqual(render.bottom, ['Y'] * 9)
        self.assertEqual(render.front, ['R'] * 9)
        self.assertEqual(render.left, ['G'] * 9)
        self.assertEqual(render.right, ['B'] * 9)


class CubeTransforms(unittest.TestCase):
    def test_top(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.T)
        r = cube3x3.render(s)
        self.assertEqual(r.top, ['W'] * 9)
        self.assertEqual(r.back, ['G'] * 3 + ['O'] * 6)
        self.assertEqual(r.bottom, ['Y'] * 9)
        self.assertEqual(r.front, ['B'] * 3 + ['R'] * 6)
        self.assertEqual(r.left, ['R'] * 3 + ['G'] * 6)
        self.assertEqual(r.right, ['O'] * 3 + ['B'] * 6)

    def test_top_rev(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.t)
        r = cube3x3.render(s)
        self.assertEqual(r.top, ['W'] * 9)
        self.assertEqual(r.back, ['B'] * 3 + ['O'] * 6)
        self.assertEqual(r.bottom, ['Y'] * 9)
        self.assertEqual(r.front, ['G'] * 3 + ['R'] * 6)
        self.assertEqual(r.left, ['O'] * 3 + ['G'] * 6)
        self.assertEqual(r.right, ['R'] * 3 + ['B'] * 6)

    def test_down(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.D)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['O'] * 6 + ['B'] * 3)
        self.assertEqual(r.bottom, ['Y'] * 9)
        self.assertEqual(r.front, ['R'] * 6 + ['G'] * 3)
        self.assertEqual(r.left, ['G'] * 6 + ['O'] * 3)
        self.assertEqual(r.right, ['B'] * 6 + ['R'] * 3)
        self.assertEqual(r.top, ['W'] * 9)

    def test_down_rev(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.d)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['O'] * 6 + ['G'] * 3)
        self.assertEqual(r.bottom, ['Y'] * 9)
        self.assertEqual(r.front, ['R'] * 6 + ['B'] * 3)
        self.assertEqual(r.left, ['G'] * 6 + ['R'] * 3)
        self.assertEqual(r.right, ['B'] * 6 + ['O'] * 3)
        self.assertEqual(r.top, ['W'] * 9)

    def test_right(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.R)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['W', 'O', 'O'] * 3)
        self.assertEqual(r.bottom, ['Y', 'Y', 'O'] * 3)
        self.assertEqual(r.front, ['R', 'R', 'Y'] * 3)
        self.assertEqual(r.left, ['G'] * 9)
        self.assertEqual(r.right, ['B'] * 9)
        self.assertEqual(r.top, ['W', 'W', 'R'] * 3)

    def test_right_rev(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.r)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['Y', 'O', 'O'] * 3)
        self.assertEqual(r.bottom, ['Y', 'Y', 'R'] * 3)
        self.assertEqual(r.front, ['R', 'R', 'W'] * 3)
        self.assertEqual(r.left, ['G'] * 9)
        self.assertEqual(r.right, ['B'] * 9)
        self.assertEqual(r.top, ['W', 'W', 'O'] * 3)

    def test_left(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.L)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['O', 'O', 'Y'] * 3)
        self.assertEqual(r.bottom, ['R', 'Y', 'Y'] * 3)
        self.assertEqual(r.front, ['W', 'R', 'R'] * 3)
        self.assertEqual(r.left, ['G'] * 9)
        self.assertEqual(r.right, ['B'] * 9)
        self.assertEqual(r.top, ['O', 'W', 'W'] * 3)

    def test_left_rev(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.l)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['O', 'O', 'W'] * 3)
        self.assertEqual(r.bottom, ['O', 'Y', 'Y'] * 3)
        self.assertEqual(r.front, ['Y', 'R', 'R'] * 3)
        self.assertEqual(r.left, ['G'] * 9)
        self.assertEqual(r.right, ['B'] * 9)
        self.assertEqual(r.top, ['R', 'W', 'W'] * 3)

    def test_front(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.F)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['O'] * 9)
        self.assertEqual(r.bottom, ['B'] * 3 + ['Y'] * 6)
        self.assertEqual(r.front, ['R'] * 9)
        self.assertEqual(r.left, ['G', 'G', 'Y'] * 3)
        self.assertEqual(r.right, ['W', 'B', 'B'] * 3)
        self.assertEqual(r.top, ['W'] * 6 + ['G'] * 3)

    def test_front_rev(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.f)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['O'] * 9)
        self.assertEqual(r.bottom, ['G'] * 3 + ['Y'] * 6)
        self.assertEqual(r.front, ['R'] * 9)
        self.assertEqual(r.left, ['G', 'G', 'W'] * 3)
        self.assertEqual(r.right, ['Y', 'B', 'B'] * 3)
        self.assertEqual(r.top, ['W'] * 6 + ['B'] * 3)

    def test_back(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.B)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['O'] * 9)
        self.assertEqual(r.bottom, ['Y'] * 6 + ['G'] * 3)
        self.assertEqual(r.front, ['R'] * 9)
        self.assertEqual(r.left, ['W', 'G', 'G'] * 3)
        self.assertEqual(r.right, ['B', 'B', 'Y'] * 3)
        self.assertEqual(r.top, ['B'] * 3 + ['W'] * 6)

    def test_back_rev(self):
        s = cube3x3.initial_state
        s = cube3x3.transform(s, cube3x3.Action.b)
        r = cube3x3.render(s)
        self.assertEqual(r.back, ['O'] * 9)
        self.assertEqual(r.bottom, ['Y'] * 6 + ['B'] * 3)
        self.assertEqual(r.front, ['R'] * 9)
        self.assertEqual(r.left, ['Y', 'G', 'G'] * 3)
        self.assertEqual(r.right, ['B', 'B', 'W'] * 3)
        self.assertEqual(r.top, ['G'] * 3 + ['W'] * 6)


if __name__ == '__main__':
    unittest.main()

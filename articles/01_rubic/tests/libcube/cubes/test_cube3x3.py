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


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import torch

from multiview3d.reconstruction import View, split_indices, triangulate_pair
from multiview3d.renderer import Camera, composite, project, quaternion_rotation, render


class ReconstructionTests(unittest.TestCase):
    def test_split_is_interleaved_and_disjoint(self):
        train, test = split_indices(20)
        self.assertEqual(test, [2, 7, 12, 17])
        self.assertEqual(len(train), 16)
        self.assertEqual(sorted(train + test), list(range(20)))

    def test_triangulation_filters_points_behind_cameras(self):
        intrinsic = np.array([[100, 0, 20], [0, 100, 20], [0, 0, 1]])
        blank = np.zeros((40, 40, 3), dtype=np.uint8)
        first = View("a", blank, intrinsic, np.eye(3), np.zeros(3))
        second = View("b", blank, intrinsic, np.eye(3), np.array([-0.3, 0, 0]))
        xyz = np.array([[0.05, -0.1, 2], [0.02, 0.02, -3]])
        points = np.column_stack((xyz, np.ones(len(xyz))))
        pixels = [points @ view.projection.T for view in (first, second)]
        left, right = (uv[:, :2] / uv[:, 2:3] for uv in pixels)
        actual, valid = triangulate_pair(first, second, left, right)
        np.testing.assert_array_equal(valid, [True, False])
        np.testing.assert_allclose(actual[valid], xyz[valid], atol=1e-7)


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(
            torch.tensor([[12., 0., 4.], [0., 12., 4.], [0., 0., 1.]]),
            torch.eye(3), torch.zeros(3), 9, 9,
        )

    def test_rotating_anisotropic_splat_rotates_projected_axes(self):
        centers = torch.tensor([[0., 0., 2.]])
        scales = torch.tensor([[[0.06, 0.2, 0.09]]])
        quaternion = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 1.]])
        factors = quaternion_rotation(quaternion) * scales
        uv, first, depth = project(self.camera, centers, factors[:1])
        _, rotated, _ = project(self.camera, centers, factors[1:])
        torch.testing.assert_close(uv, torch.tensor([[4., 4.]]))
        torch.testing.assert_close(depth, torch.tensor([2.]))
        torch.testing.assert_close(first[0].diag().flip(0), rotated[0].diag())

    def test_front_to_back_compositing_resets_at_each_pixel(self):
        actual = composite(
            torch.tensor([0, 0, 1]), torch.tensor([0.5, 0.25, 0.25]),
            torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]),
            torch.tensor([0., 0., 0.]), 3,
        )
        expected = torch.tensor([[0.5, 0., 0.125], [0., 0.25, 0.], [0., 0., 0.]])
        torch.testing.assert_close(actual, expected)

    def test_visible_splat_has_finite_center_scale_color_and_alpha_gradients(self):
        centers = torch.tensor([[0.05, 0., 2.], [0., 0., -1.]], requires_grad=True)
        factors = (torch.eye(3).repeat(2, 1, 1) * 0.15).requires_grad_()
        rgb = torch.tensor([[0.8, 0.3, 0.2], [0., 1., 0.]], requires_grad=True)
        opacity = torch.tensor([0.75, 0.99], requires_grad=True)
        background = torch.tensor([0.05, 0.05, 0.05])
        image = render(self.camera, centers, factors, rgb, opacity, background)
        self.assertEqual(tuple(image.shape), (9, 9, 3))
        self.assertGreater(float(image[4, 4, 0].detach()), 0.5)
        image.square().mean().backward()
        for parameter in (centers, factors, rgb, opacity):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad[0].abs().max()), 0)
            self.assertEqual(float(parameter.grad[1].abs().max()), 0)


if __name__ == "__main__":
    unittest.main()

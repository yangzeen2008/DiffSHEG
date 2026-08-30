import unittest
from types import SimpleNamespace

import numpy as np
import torch

import datasets.rotation_converter as rot_cvt
from assets.analyze_full_validation_transitions import summarize_expressions
from models.flow_matching import FlowMatching
from smooth_adaptive import smooth_euler_rotation
from utils.motion_metrics import linear_pck_mse, rotation_pck_mse, select_axis_angle_stats
from utils.window_stitching import (
    audio_windows_are_consecutive,
    make_prefix_inpaint,
    resolve_overlap_len,
)
from utils.test_selection import IndexedDatasetView, parse_test_window_ranges, safe_result_tag


class FlowTransitionTests(unittest.TestCase):
    def test_release_blend_matches_known_velocity(self):
        flow = FlowMatching(SimpleNamespace(
            fm_sample_steps=1,
            fm_solver='euler',
            fm_transition_blend=3,
        ))
        gt = torch.zeros(1, 8, 2)
        gt[:, 0] = 0.0
        gt[:, 1] = 1.0
        gt[:, 2] = 2.0
        sample = torch.full_like(gt, 100.0)
        sample[:, :3] = gt[:, :3]
        mask = torch.zeros_like(gt, dtype=torch.bool)
        mask[:, :3] = True

        blended = flow._blend_inpaint_release(sample, gt, mask)

        torch.testing.assert_close(blended[:, :3], gt[:, :3])
        torch.testing.assert_close(blended[:, 3], torch.full((1, 2), 3.0))
        torch.testing.assert_close(blended[:, 5], sample[:, 5])
        self.assertTrue((blended[:, 3:6] >= 3.0).all())
        self.assertTrue((blended[:, 3:6] <= 100.0).all())

    def test_release_blend_preserves_linear_continuation(self):
        flow = FlowMatching(SimpleNamespace(
            fm_sample_steps=1,
            fm_solver='euler',
            fm_transition_blend=4,
        ))
        gt = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
        sample = gt.clone()
        mask = torch.zeros_like(gt, dtype=torch.bool)
        mask[:, :3] = True

        blended = flow._blend_inpaint_release(sample, gt, mask)

        torch.testing.assert_close(blended, sample)

    def test_axis_angle_release_uses_short_so3_path_across_pi(self):
        flow = FlowMatching(SimpleNamespace(
            fm_sample_steps=1,
            fm_solver='euler',
            fm_transition_blend=7,
        ))
        flow.set_axis_angle_stats(
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            gesture_dim=3,
        )

        sample = torch.zeros(1, 12, 4)
        gt = torch.zeros_like(sample)
        known_degrees = (169.0, 174.0, 179.0)
        for frame, angle in enumerate(known_degrees):
            value = np.deg2rad(angle)
            sample[0, frame, 0] = value
            gt[0, frame, 0] = value
            sample[0, frame, 3] = float(frame)
            gt[0, frame, 3] = float(frame)

        # The physical target is only ten degrees beyond the extrapolated
        # first frame, but its principal axis-angle vector has the opposite
        # sign. Euclidean Hermite interpolation would rotate almost 360 deg.
        sample[0, 3:10, 0] = torch.linspace(
            np.deg2rad(-176.0), np.deg2rad(-171.0), 7
        )
        sample[0, 10:, 0] = torch.tensor([
            np.deg2rad(-166.0), np.deg2rad(-161.0)
        ])
        sample[0, 3:, 3] = 100.0
        mask = torch.zeros_like(sample, dtype=torch.bool)
        mask[:, :3] = True

        blended = flow._blend_inpaint_release(sample, gt, mask)

        torch.testing.assert_close(blended[:, :3], gt[:, :3])
        rotations = rot_cvt.axis_angle_to_matrix(blended[0, 3:10, :3])
        relative = rotations[:-1].transpose(-1, -2) @ rotations[1:]
        steps = torch.linalg.vector_norm(
            rot_cvt.matrix_to_axis_angle(relative), dim=-1
        )
        endpoint = torch.linalg.vector_norm(
            rot_cvt.matrix_to_axis_angle(
                rotations[0].transpose(-1, -2) @ rotations[-1]
            )
        )
        self.assertLessEqual(steps.sum().item(), endpoint.item() + 1e-5)
        self.assertLess(np.rad2deg(steps.max().item()), 5.0)
        self.assertTrue((blended[0, 3:10, 3] >= 3.0).all())
        self.assertTrue((blended[0, 3:10, 3] <= 100.0).all())


class WindowStitchingTests(unittest.TestCase):
    def test_auto_overlap_matches_training_stride(self):
        self.assertEqual(
            resolve_overlap_len(34, 10, 0, auto_overlap=True, sequential=True),
            24,
        )
        self.assertEqual(
            resolve_overlap_len(34, 10, 0, auto_overlap=True, sequential=False),
            0,
        )

    def test_consecutive_audio_windows_are_detected(self):
        source = torch.arange(50000, dtype=torch.float32)
        previous = source[:36266]
        current = source[10666:10666 + 36266]
        unrelated = -current
        self.assertTrue(audio_windows_are_consecutive(
            previous,
            current,
            pose_stride=10,
            pose_fps=15,
            audio_fps=16000,
        ))
        self.assertFalse(audio_windows_are_consecutive(
            previous,
            unrelated,
            pose_stride=10,
            pose_fps=15,
            audio_fps=16000,
        ))

    def test_prefix_inpaint_reuses_previous_tail_exactly(self):
        template = torch.zeros(1, 34, 3)
        previous = torch.arange(34 * 3, dtype=torch.float32).reshape(1, 34, 3)
        request = make_prefix_inpaint(template, previous, 24)
        torch.testing.assert_close(request['gt'][:, :24], previous[:, -24:])
        self.assertTrue(request['outpainting_mask'][:, :24].all())
        self.assertFalse(request['outpainting_mask'][:, 24:].any())

    def test_bounded_test_ranges_preserve_segment_order(self):
        indices = parse_test_window_ranges("12:3,2:2")
        self.assertEqual(indices, [12, 13, 14, 2, 3])
        view = IndexedDatasetView(list(range(20)), indices)
        self.assertEqual([view[i] for i in range(len(view))], indices)

    def test_test_range_validation_and_safe_tag(self):
        with self.assertRaises(ValueError):
            parse_test_window_ranges("5:0")
        with self.assertRaises(ValueError):
            parse_test_window_ranges("5:2,6:2")
        self.assertEqual(safe_result_tag(" long 3 / diverse "), "long_3_diverse")


class RotationSmoothingTests(unittest.TestCase):
    def test_quaternion_smoothing_does_not_average_wrap_to_zero(self):
        values = np.array([
            [170.0, 0.0, 0.0],
            [175.0, 0.0, 0.0],
            [179.0, 0.0, 0.0],
            [-179.0, 0.0, 0.0],
            [-175.0, 0.0, 0.0],
        ])
        smoothed = smooth_euler_rotation(values, 'XYZ', sigma=1.0)
        self.assertGreater(abs(smoothed[2, 0]), 150.0)
        self.assertLess(np.abs(np.diff(smoothed[:, 0])).max(), 20.0)


class MotionMetricTests(unittest.TestCase):
    def test_facial_coefficients_are_excluded(self):
        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        target = torch.zeros(2, 4, 9)
        prediction = target.clone()
        prediction[..., 6:] = 1000.0

        pck, mse, _ = rotation_pck_mse(
            prediction,
            target,
            threshold=0.5,
            rot_6d=False,
            split_pos=6,
            mean_axis_angle=mean,
            std_axis_angle=std,
        )
        self.assertEqual(pck, 1.0)
        self.assertAlmostEqual(mse, 0.0, places=7)

    def test_6d_and_axis_angle_identity_have_zero_error(self):
        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        axis_angle = torch.zeros(1, 3, 2, 3)
        rotation_6d = rot_cvt.axis_angle_to_rotation_6d(axis_angle).reshape(1, 3, 12)
        pck, mse, _ = rotation_pck_mse(
            rotation_6d,
            rotation_6d.clone(),
            threshold=0.5,
            rot_6d=True,
            split_pos=12,
            mean_axis_angle=mean,
            std_axis_angle=std,
        )
        self.assertEqual(pck, 1.0)
        self.assertAlmostEqual(mse, 0.0, places=7)

    def test_expression_metrics_are_coefficient_wise(self):
        target = torch.zeros(1, 2, 51)
        prediction = target.clone()
        prediction[..., 0] = 1.0
        pck, mse, errors = linear_pck_mse(prediction, target, threshold=0.5)
        self.assertEqual(errors.shape, target.shape)
        self.assertAlmostEqual(pck, 50 / 51)
        self.assertAlmostEqual(mse, 1 / 51)

    def test_expression_report_separates_tiny_and_significant_range_errors(self):
        target = np.zeros((44, 51), dtype=np.float32)
        prediction = target.copy()
        prediction[:, 0] = -0.005
        prediction[:, 1] = -0.02
        boundary = [{
            "chain_index": 0,
            "frame": 34,
            "output_window_index": 1,
        }]

        report = summarize_expressions(
            [prediction], [target], boundary, boundary
        )

        self.assertGreater(report["range"]["raw_out_of_0_1_rate"], 0.0)
        self.assertGreater(report["range"]["significant_out_of_0_1_rate"], 0.0)
        self.assertLess(
            report["range"]["significant_out_of_0_1_rate"],
            report["range"]["raw_out_of_0_1_rate"],
        )
        self.assertEqual(report["boundary"]["abnormal_rate"], 0.0)

    def test_remove_hand_selects_training_indices(self):
        mean = np.arange(141)
        std = mean + 1
        selected_mean, selected_std = select_axis_angle_stats(mean, std, 33)
        expected = np.array(list(range(21)) + list(range(75, 87)))
        np.testing.assert_array_equal(selected_mean, expected)
        np.testing.assert_array_equal(selected_std, expected + 1)


if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest

import numpy as np

from scripts.adaptive_trust import (OnlineReliability,
                                    degrade_unknown_geometry, fuse_scores)


class ReliabilityTests(unittest.TestCase):
    def test_only_future_new_observations_are_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            estimator = OnlineReliability(0.5, 2.0, 1.0, 2, directory)
            prediction = np.array([[0.2, 0.8], [0.7, 0.4]])
            observed = np.array([[0.0, 0.5], [0.5, 0.5]])
            estimator.capture(0, prediction, observed, padding=[1, 2, 3, 4])
            later = np.array([[0.0, 1.0], [0.0, 0.5]])
            first = estimator.observe(1, later)
            self.assertEqual(first['new_cells'], 2)
            self.assertTrue(first['updated'])
            self.assertAlmostEqual(first['update_error'], (0.2 + 0.7) / 2)
            self.assertLess(first['reliability'], 1.0)
            repeated = estimator.observe(2, later)
            self.assertEqual(repeated['new_cells'], 0)
            self.assertFalse(repeated['updated'])
            self.assertEqual(repeated['reliability'], first['reliability'])
            snapshot = np.load(directory + '/prediction_00000000.npz')
            self.assertTrue(np.array_equal(snapshot['padding'], [1, 2, 3, 4]))

    def test_small_batches_accumulate_without_false_update(self):
        estimator = OnlineReliability(0.8, 1.0, 1.0, 3)
        observed = np.full((2, 2), 0.5)
        estimator.capture(0, np.zeros((2, 2)), observed)
        later = observed.copy()
        later[0, 0] = 1
        self.assertFalse(estimator.observe(1, later)['updated'])
        self.assertFalse(estimator.observe(2, later)['updated'])
        later[0, 1] = 1
        later[1, 0] = 1
        self.assertTrue(estimator.observe(3, later)['updated'])

    def test_shape_and_probability_checks(self):
        estimator = OnlineReliability(0.5, 1.0, 1.0, 1)
        with self.assertRaises(ValueError):
            estimator.capture(0, np.zeros((2, 3)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            estimator.capture(0, np.full((2, 2), 2.0), np.zeros((2, 2)))
        estimator.capture(0, np.zeros((2, 2)), np.full((2, 2), 0.5))
        with self.assertRaises(ValueError):
            estimator.observe(1, np.zeros((3, 3)))

    def test_replanning_does_not_recount_already_seen_cells(self):
        estimator = OnlineReliability(0.5, 1.0, 1.0, 1)
        initial = np.full((2, 2), 0.5)
        estimator.capture(0, np.zeros((2, 2)), initial)
        first_observation = initial.copy()
        first_observation[0, 0] = 1
        self.assertEqual(estimator.observe(1, first_observation)['new_cells'], 1)
        estimator.capture(1, np.zeros((2, 2)), first_observation)
        self.assertEqual(estimator.observe(2, first_observation)['new_cells'], 0)
        later = first_observation.copy()
        later[1, 1] = 1
        self.assertEqual(estimator.observe(3, later)['new_cells'], 1)


class FusionTests(unittest.TestCase):
    def test_endpoints_and_interior(self):
        pipe = [8.0, 5.0]
        classical = [2.0, 9.0]
        self.assertEqual(fuse_scores(pipe, classical, 1.0, 0, 1)[0], 0)
        self.assertEqual(fuse_scores(pipe, classical, 0.0, 0, 1)[0], 1)
        self.assertEqual(fuse_scores(pipe, classical, 0.5)[0], 1)

    def test_zero_and_nonfinite_scores(self):
        best, fused, _, _ = fuse_scores([0, np.nan], [0, 0], 0.5)
        self.assertEqual(best, 0)
        self.assertTrue(np.all(np.isfinite(fused)))
        self.assertTrue(np.array_equal(fused, [0, 0]))


class DegradationTests(unittest.TestCase):
    def test_only_unknown_prediction_is_corrupted(self):
        prediction = np.array([[0.2, 0.8], [0.1, 0.9]])
        observed = np.array([[0, 0.5], [1, 0.5]])
        degraded = degrade_unknown_geometry(prediction, observed, 0.75)
        self.assertAlmostEqual(degraded[0, 0], 0.2)
        self.assertAlmostEqual(degraded[1, 0], 0.1)
        self.assertAlmostEqual(degraded[0, 1], 0.35)
        self.assertAlmostEqual(degraded[1, 1], 0.3)


if __name__ == '__main__':
    unittest.main()

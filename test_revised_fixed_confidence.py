import math
import unittest

import numpy as np

from revised_fixed_confidence import (
    clopper_pearson_upper,
    hypothesis_means,
    optimal_design,
    update_exact_scores,
    verified_decision,
)


class RevisedProtocolTests(unittest.TestCase):
    def test_exact_likelihood_prefers_matching_mean(self):
        sigma = np.eye(2)
        means = np.array([[1.0, 0.0], [0.0, 1.0]])
        scores = np.zeros(2)
        c = np.array([1.0, 0.0])
        update_exact_scores(scores, c, 1.0, means, sigma)
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(verified_decision(scores, 0.49), 0)
        self.assertIsNone(verified_decision(scores, 0.51))

    def test_sdp_support_matches_rate(self):
        _, means = hypothesis_means(3, 1, 1.0)
        design = optimal_design(np.eye(3), means, 0, 2.0)
        self.assertAlmostEqual(float(design["probabilities"].sum()), 1.0, places=10)
        self.assertLess(design["rate_error"], 5e-4)
        for action in design["actions"]:
            self.assertAlmostEqual(float(np.linalg.norm(action, 1)), 2.0, places=8)

    def test_zero_error_upper_bound(self):
        self.assertAlmostEqual(clopper_pearson_upper(0, 40), 1 - 0.05 ** (1 / 40), places=12)
        self.assertTrue(math.isclose(clopper_pearson_upper(40, 40), 1.0))


if __name__ == "__main__":
    unittest.main()

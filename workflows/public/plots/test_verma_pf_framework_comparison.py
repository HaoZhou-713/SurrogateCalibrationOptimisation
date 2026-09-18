"""Numerical checks for saved-front summaries; no optimisation is run."""
import importlib
import unittest

import numpy as np
from scipy.spatial import ConvexHull


class FrontSummaryTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module("plot_verma_pf_framework_comparison")

    def test_hausdorff_is_symmetric_and_uses_most_distant_point(self):
        a = np.array([[0., 0, 0], [2, 0, 0]])
        b = np.array([[0., 0, 0]])
        self.assertEqual(self.m.symmetric_hausdorff(a, b), 2.)
        self.assertEqual(self.m.symmetric_hausdorff(b, a), 2.)

    def test_medoid_is_actual_central_run_and_dispersion_excludes_diagonal(self):
        fronts = [np.array([[x, 0., 0.]]) for x in [0., 1., 4.]]
        index, dispersion, distances = self.m.select_medoid(fronts)
        self.assertEqual(index, 1)
        self.assertAlmostEqual(dispersion, 8. / 3.)
        np.testing.assert_allclose(distances, [[0, 1, 4], [1, 0, 3], [4, 3, 0]])

    def test_support_distance_handles_inside_faces_edges_and_vertices(self):
        hull = ConvexHull(np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        queries = np.array([[.1, .1, .1], [1, 1, 1], [-1, -1, 0], [2, 0, 0], [.5, .5, 0]])
        distances = self.m.distance_to_support(queries, hull)
        np.testing.assert_allclose(distances, [0, 2 / np.sqrt(3), np.sqrt(2), 1, 0], atol=1e-12)

    def test_single_run_has_undefined_dispersion(self):
        index, dispersion, matrix = self.m.select_medoid([np.array([[0., 0., 0.]])])
        self.assertEqual(index, 0)
        self.assertTrue(np.isnan(dispersion))
        self.assertEqual(matrix.shape, (1, 1))

    def test_background_and_medoid_retain_points_dominated_in_projection(self):
        # Second point is worse in T/P but better in TD: both must be plotted.
        points = np.array([[0., 0, 2], [1, 1, 0]])
        run = self.m.FrontRun(4, np.zeros((2, 4)), points)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        self.m.draw_fronts(ax, [run], "raw", (0, 1), representative=run)
        self.assertEqual(len(ax.collections), 2)
        for collection in ax.collections:
            np.testing.assert_array_equal(collection.get_offsets(), points[:, [0, 1]])
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()

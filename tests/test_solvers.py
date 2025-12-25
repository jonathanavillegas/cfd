"""
Unit tests for solver modules.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import unittest
from cfd.solvers.poisson_1d import solve_1d_steady
from cfd.solvers.poisson_2d import solve_2d_steady


class TestSolvers(unittest.TestCase):
    """Test cases for solver functions."""
    
    def test_1d_steady(self):
        """Test 1D steady-state solver."""
        x, T = solve_1d_steady(num_nodes=10, length=1.0, k=1.0, BCR_type="dirichlet")
        self.assertEqual(len(x), 10)
        self.assertEqual(len(T), 10)
        self.assertTrue(np.all(np.isfinite(T)))
    
    def test_2d_steady(self):
        """Test 2D steady-state solver."""
        x, y, T = solve_2d_steady(numXNodes=10, numYNodes=10, xL=1.0, yL=1.0, k=1.0)
        self.assertEqual(len(x), 10)
        self.assertEqual(len(y), 10)
        self.assertEqual(T.shape, (10, 10))
        self.assertTrue(np.all(np.isfinite(T)))


if __name__ == '__main__':
    unittest.main()


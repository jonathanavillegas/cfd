"""
Unit tests for boundary condition utilities.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import unittest
from cfd.utils.boundary_conditions import (
    boundary_matrix, set_dirichlet, set_neumann, def_BCSolution
)


class TestBoundaryConditions(unittest.TestCase):
    """Test cases for boundary condition functions."""
    
    def test_boundary_matrix(self):
        """Test boundary matrix generation."""
        BW, BE, BN, BS = boundary_matrix(25, 5, 5)
        self.assertEqual(len(BW), 5)
        self.assertEqual(len(BE), 5)
        self.assertEqual(len(BN), 5)
        self.assertEqual(len(BS), 5)
    
    def test_set_dirichlet(self):
        """Test Dirichlet BC setup."""
        BC = set_dirichlet(10, 5)
        self.assertEqual(BC[5], 1)
        self.assertEqual(np.sum(BC), 1)
    
    def test_def_BCSolution(self):
        """Test BC solution calculation."""
        # Dirichlet
        result = def_BCSolution("dirichlet", 0, 0.1, 1.0, 0)
        self.assertEqual(result, 0)
        
        # Neumann
        result = def_BCSolution("neumann", 0, 0.1, 1.0, 5)
        self.assertAlmostEqual(result, 1000.0, places=1)


if __name__ == '__main__':
    unittest.main()


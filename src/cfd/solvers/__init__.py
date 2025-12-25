"""
Solver modules for various heat transfer problems.
"""
from .poisson_1d import solve_1d_steady
from .poisson_2d import solve_2d_steady
from .transient_2d import solve_2d_transient
from .heat_shield import solve_heat_shield_2d, solve_heat_shield_3d


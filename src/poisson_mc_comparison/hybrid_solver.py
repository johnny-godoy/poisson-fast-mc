"""Implement a Hybrid solver that combines the Monte Carlo solver and the Finite Difference solver."""
from __future__ import annotations

import numpy as np
import scipy as sp
import scipy.sparse.linalg

from finite_difference_solver import BaseFiniteDifferenceSolver
from subwalk_monte_carlo_solvers import MultiThreadSMCSolver


class HybridSolver(BaseFiniteDifferenceSolver):
    """A solver that combines Monte Carlo estimations with Finite Difference approximations."""
    __slots__ = ()

    def __init__(self, problem: PoissonProblem, random_walks: int):
        super().__init__(problem)
        # Consiguiendo la heurística
        x0 = MultiThreadSMCSolver(problem, random_walks).grid[1:-1, 1:-1].flatten()
        interior = np.array(sp.sparse.linalg.cg(self.A, self.b, x0=x0)[0]).reshape(self.N, self.N)
        self.grid[1:-1, 1:-1] = interior

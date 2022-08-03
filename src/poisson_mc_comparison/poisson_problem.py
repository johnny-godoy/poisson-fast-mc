"""Implement the PoissonProblem class that contains the problem definition."""
from __future__ import annotations

import numpy as np


class PoissonProblem:
    """Hold a Poisson problem in a square domain with Dirichlet boundary conditions.
    Attributes:
    -----------
    discretization: int
        The number of discretization points.
    h: float
        The step size.
    side: np.ndarray
        A discretized side of the square.
    init_grid: np.ndarray
        The initial solution of the Poisson problem.
    f: callable
        The discretized function that defines the right hand side of the Poisson problem.
    g: callable
        The discretized function that defines the Dirichlet boundary condition of the Poisson problem."""
    __slots__ = 'discretization', 'h', 'side', 'init_grid', 'f', 'g'

    def __init__(self, discretization: int, f: callable = None, g: callable = None):
        """
        Parameters
        ----------
        discretization: int
            The number of discretization points.
        f: callable, optional
            The discretized function that defines the right hand side of the Poisson problem.
        g: callable, optional
            The discretized function that defines the Dirichlet boundary condition of the Poisson problem."""
        self.discretization = discretization
        self.h = 1/(self.discretization + 1)
        self.side = np.linspace(0, 1, self.discretization + 2)
        self.init_grid = np.zeros((self.discretization + 2, self.discretization + 2))

        if f is None:
            self.f = lambda i, _: np.zeros_like(self.side[i])

        if g is None:
            self.g = lambda i, j: np.where(i == self.discretization + 1,
                                           np.sin(np.pi*self.side[j]), 0)

        zeros = np.zeros_like(self.side, dtype=int)
        Ns = self.discretization*np.ones_like(self.side, dtype=int) + 1
        side_range = np.arange(self.discretization + 2)
        self.init_grid[0] = self.g(zeros, side_range)
        self.init_grid[-1] = self.g(Ns, side_range)
        self.init_grid[:, 0] = self.g(side_range, zeros)
        self.init_grid[:, -1] = self.g(side_range, Ns)

    def __repr__(self):
        return f"{self.__class__.__name__}(discretization={self.discretization})"

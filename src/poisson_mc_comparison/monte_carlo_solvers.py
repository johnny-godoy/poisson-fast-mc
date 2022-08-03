"""Implement solver from the algorithm here https://github.com/s-ankur/montecarlo-pde, with Bayesian estimators."""
from __future__ import annotations

from contextlib import suppress
import abc
import dataclasses

import numpy as np

from _utilities import get_random_walk


@dataclasses.dataclass
class BaseSolver:
    """A base Monte Carlo solver for the Poisson problem.
    Parameters
    ----------
    problem: PoissonProblem
        The object containing the problem definition.
    random_walks: int
        The number of random walks to use.
    Attributes
    ----------
    borders: np.ndarray
        An array that contains the indexes of the border points of the domain.
    grid: np.ndarray
        The grid that stores the final solution.
    g_vals: np.ndarray
        An array that contains the values of the function g."""
    __slots__ = 'problem', 'random_walks', 'borders', 'grid', 'g_vals'
    problem: PoissonProblem
    random_walks: int

    def __post_init__(self):
        self.grid = self.problem.init_grid.copy()
        N = self.problem.discretization

        zeros = np.zeros(N, dtype=int)
        edges = (N + 1) * np.ones(N, dtype=int)
        inner = np.arange(1, N + 1)
        self.borders = np.vstack((np.column_stack((zeros, inner)), np.column_stack((edges, inner)), np.column_stack((inner, zeros)),
                                  np.column_stack((inner, edges))))
        self.g_vals = self.problem.g(self.borders.T[1], self.borders.T[0])

        X, Y = np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1))
        self.grid[1:-1, 1:-1] = np.vectorize(self.u_pointwise)(X, Y)

    def g(self, i, j):
        """Returns the value of the function g at the point (i, j)."""
        return self.problem.init_grid[i, j]

    def get_walk_information(self, i, j):
        """Simulates many random walks from (i, j) and returns the information necessary
        to estimate the value of the solution at that point.
        Parameters
        ----------
        i: int
            The row index of the starting point.
        j: int
            The column index of the starting point.
        Returns
        -------
        walk_ends: np.ndarray of size (random_walks, 2)
            An array that contains the indexes of the endpoint for each walk.
        F: np.ndarray of size (random_walks, )
            An array that contains the right hand side of the Poisson problem summed for all interior points of each walk."""
        F = np.zeros(self.random_walks)
        walk_ends = np.zeros((self.random_walks, 2), dtype=int)
        for k in range(self.random_walks):
            i_walk, j_walk, walk = get_random_walk(i, j, self.problem.discretization + 1)
            walk_ends[k] = (i_walk, j_walk)
            F[k] = self.problem.f(*np.array(walk).T).sum()
        return walk_ends, F

    def prior_frequencies(self, i, j):
        """Returns the prior frequencies for reaching each endpoint when starting from (i, j).
        Parameters
        ----------
        i: int
            The row index of the starting point.
        j: int
            The column index of the starting point.
        Returns
        -------
        prior: np.ndarray
            An array that contains the prior frequencies for each endpoint."""
        raise NotImplementedError

    def posterior_frequencies(self, i, j, walk_ends):
        """Returns the posterior frequencies for reaching each endpoint when starting from (i, j).
         Parameters
         ----------
         i: int
             The row index of the starting point.
         j: int
             The column index of the starting point.
         walk_ends: np.ndarray
             An array that contains the indexes of the endpoint for each walk.
         Returns
         -------
         posterior: np.ndarray
             An array that contains the posterior frequencies for each endpoint."""
        posterior = self.prior_frequencies(i, j)
        values, frequencies = np.unique(walk_ends, axis=0, return_counts=True)
        for i, border in enumerate(self.borders):
            border_indexes = np.argwhere(np.all(border == values, axis=1))
            with suppress(IndexError):
                border_index = border_indexes[0]
                posterior[i] += frequencies[border_index]
        return posterior

    def u_pointwise(self, i, j):
        """Returns the value of the solution at the point (i, j).
        Parameters
        ----------
        i: int
            The row index of the point.
        j: int
            The column index of the point.
        Returns
        -------
        u: float
            The estimated value of the solution at the point (i, j)."""
        walk_end, F = self.get_walk_information(i, j)
        posterior = self.posterior_frequencies(i, j, walk_end)
        g_s = posterior.dot(self.g_vals) / np.sum(posterior)
        u = g_s - F.sum() * self.problem.h * self.problem.h
        return u


class FrequentistSolver(BaseSolver, abc.ABC):
    """Monte Carlo solver for the Poisson problem that uses a frequentist probability estimation."""
    __slots__ = ()

    def prior_frequencies(self, _, __):
        return np.zeros(len(self.borders))


class LaplaceSmoothingSolver(BaseSolver, abc.ABC):
    """Monte Carlo solver for the Poisson problem that uses Laplace smoothing."""
    __slots__ = ()

    def prior_frequencies(self, _, __):
        return 0.5 * np.ones(len(self.borders)) / self.g_vals.sum()


class ManhattanSmoothingSolver(BaseSolver, abc.ABC):
    """Monte Carlo solver for the Poisson problem that uses smoothing based on the reciprocal of the Manhattan distance."""
    __slots__ = ()

    def prior_frequencies(self, i, j):
        return 1 / np.abs(self.borders - np.array([i, j])).sum(axis=1)

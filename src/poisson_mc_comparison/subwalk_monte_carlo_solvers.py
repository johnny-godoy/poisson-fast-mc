"""Implement our custom solvers that use the generated subwalks for speed improvement."""
from __future__ import annotations

import dataclasses
import multiprocessing.pool

import numpy as np

from _utilities import get_random_walk


@dataclasses.dataclass
class BaseSMCSolver:
    """A base class for Subwalk Monte Carlo Poisson Problem solvers.
    Parameters
    ----------
    problem: PoissonProblem
        The object containing the problem definition.
    random_walks: int
        The number of random walks to use.
    Attributes
    ----------
    borders: np.ndarray
        An array that contains the indexes of the points in the border of the grid.
    borders_val: np.ndarray
        An array that contains the values of the border function.
    F: np.ndarray
        An array that contains the values of the right hand side of the Poisson problem scaled by the negative step size.
    walks_through_point: np.ndarray
        An array that contains the summed amount each point in the grid is visited.
    border_contribution: np.ndarray
        An array that contains the summed amount all borders contribute to the grid.
    grid: np.ndarray
        The grid that stores the final solution."""
    __slots__ = 'problem', 'random_walks', 'borders', 'borders_val', 'F', 'walks_through_point', 'border_contribution', 'grid'
    problem: PoissonProblem
    random_walks: int

    def __post_init__(self):
        zeros = np.zeros(self.problem.discretization, dtype=int)
        edges = (self.problem.discretization + 1) * np.ones(self.problem.discretization, dtype=int)
        inner = np.arange(1, self.problem.discretization + 1, dtype=int)
        self.borders = np.vstack((np.column_stack((zeros, inner)), np.column_stack((edges, inner)), np.column_stack((inner, zeros)),
                                  np.column_stack((inner, edges))))
        self.borders_val = self.g(*self.borders.T)

        self.F = np.zeros_like(self.problem.init_grid)
        self.F[1:-1, 1:-1] = -self.problem.f(np.arange(1, self.problem.discretization + 1),
                                             np.arange(1, self.problem.discretization + 1)) * self.problem.h * self.problem.h

        self.walks_through_point = np.ones_like(self.problem.init_grid) / (2 * self.borders_val.sum())
        self.border_contribution = np.ones_like(self.problem.init_grid) / 2
        self.grid = self.problem.init_grid.copy()

    def g(self, i, j):
        """Returns the value of the border function at the point (i, j)."""
        return self.problem.init_grid[i, j]

    def process_walk(self, indexes):
        """Processes a random walk.
        Parameters
        ----------
        indexes: np.ndarray
            The indexes of the points in the random walk.
        Returns
        -------
        grid: np.ndarray
            The grid updated with the right hand side of the Poisson problem.
        border_contribution: np.ndarray
            The summed amount all borders contribute to the grid.
        walks_through_point: np.ndarray
            The summed amount each point in the grid is visited."""
        grid = np.zeros_like(self.problem.init_grid)
        walks_through_point = np.zeros_like(self.problem.init_grid)
        border_contribution = np.zeros_like(self.problem.init_grid)
        for initial_point in indexes:
            i_walk, j_walk, walk = get_random_walk(*initial_point, self.problem.discretization + 1)
            border_val = self.g(i_walk, j_walk)
            reverse_walk = np.array(walk)[::-1]
            for step, F in zip(reverse_walk, np.cumsum(self.F[reverse_walk])):
                x, y = step
                grid[x, y] += F
                border_contribution[x, y] += border_val
                walks_through_point[x, y] += 1
            antithetic_walk = self.problem.discretization + 1 - reverse_walk
            border_val = self.g(self.problem.discretization + 1 - i_walk, self.problem.discretization + 1 - j_walk)
            for step, F in zip(antithetic_walk, np.cumsum(self.F[antithetic_walk])):
                x, y = step
                grid[x, y] += F
                border_contribution[x, y] += border_val
                walks_through_point[x, y] += 1
        return grid, border_contribution, walks_through_point


class SingleThreadSMCSolver(BaseSMCSolver):
    """A class for solving the Poisson Problem with a single threaded Subwalk Monte Carlo implementation."""
    __slots__ = ()

    def __init__(self, problem: PoissonProblem, random_walks: int):
        super().__init__(problem, random_walks)
        grid, border_contribution, walks_through_point = self.process_walk(
            np.random.randint(1, self.problem.discretization + 1, size=(self.random_walks, 2)))
        self.walks_through_point = walks_through_point
        self.border_contribution = border_contribution
        self.grid += grid
        self.grid[1:-1, 1:-1] += (self.border_contribution / self.walks_through_point)[1:-1, 1:-1]


class MultiThreadSMCSolver(BaseSMCSolver):
    """A class for solving the Poisson Problem with a multi threaded Subwalk Monte Carlo implementation."""
    __slots__ = ()

    def __init__(self, problem: PoissonProblem, random_walks: int):
        super().__init__(problem, random_walks)

        r = self.problem.discretization // 2
        rs = r * np.ones(self.random_walks, dtype=int)
        zeros = np.zeros(self.random_walks, dtype=int)
        size = (self.random_walks, 2)
        random_states = (np.random.randint(1, r + 1, size=size), np.random.randint(1, r + 1, size=size) + np.column_stack((rs, zeros)),
                         np.random.randint(1, r + 1, size=size) + np.column_stack((zeros, rs)),
                         np.random.randint(1, r + 1, size=size) + np.column_stack((rs, rs)))

        pool = multiprocessing.pool.ThreadPool(processes=4)
        r = np.array(pool.map(self.process_walk, random_states))
        pool.close()
        pool.join()

        self.grid += np.sum(r[:, 0], axis=0)
        self.border_contribution += np.sum(r[:, 1], axis=0)
        self.walks_through_point += np.sum(r[:, 2], axis=0)
        self.grid[1:-1, 1:-1] += (self.border_contribution / self.walks_through_point)[1:-1, 1:-1]

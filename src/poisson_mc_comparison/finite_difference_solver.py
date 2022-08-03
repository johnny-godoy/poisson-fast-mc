"""Implement a finite difference solver for the Poisson equation."""
from __future__ import annotations

import dataclasses

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg


@dataclasses.dataclass
class BaseFiniteDifferenceSolver:
    """Base class for finite difference solvers.
    Parameters
    ----------
    problem: PoissonProblem
        The object containing the problem definition.
    Attributes
    ----------
    N: int
        The number of discretization points.
    A: scipy.sparse.csc_matrix
        The matrix that stores the finite difference operator.
    b: np.ndarray
        An array that contains the right hand side of the Poisson problem.
    grid: np.ndarray
        The grid that stores the final solution."""
    __slots__ = 'problem', 'N', 'A', 'b', 'grid'
    problem: PoissonProblem

    def __post_init__(self):
        N = self.problem.discretization

        h = self.problem.h
        self.N = N

        # Calculando A
        neighbours = -np.ones(N - 1)
        # Creando la diagonal principal de la matriz
        L_4 = sp.sparse.diags([neighbours, 4*np.ones(N), neighbours], [-1, 0, 1])
        # Creando las diagonales vecinas
        LR = sp.sparse.diags([neighbours, neighbours], [-1, 1])
        # Insertándolas por bloques a través de la suma de Kronecker
        A = sp.sparse.kronsum(L_4, LR, format='csc')  # Equivalente a kron(L_4, np.eye(N)) + kron(np.eye(N), LR)
        self.A = A / (h * h)

        # Calculando b
        grid = self.problem.init_grid
        X, Y = np.meshgrid(np.arange(N), np.arange(N))
        self.b = self.problem.f(X, Y).flatten()
        self.b[:N] += grid[0, 1:-1] / (h * h)
        self.b[-N:] += grid[-1, 1:-1] / (h * h)
        indexes = np.arange(0, N * N, N)
        self.b[indexes] += grid[1:-1, 0] / (h * h)
        self.b[indexes + N - 1] += grid[1:-1, -1] / (h * h)
        self.grid = grid.copy()


class FiniteDifferenceSolver(BaseFiniteDifferenceSolver):
    """A finite difference solver for the Poisson problem."""
    __slots__ = ()

    def __init__(self, problem: PoissonProblem):
        super().__init__(problem)
        # Resolviendo el sistema
        interior_vector = sp.sparse.linalg.spsolve(self.A, self.b)
        interior = interior_vector.reshape((self.N, self.N)).T
        # Rellenando los valores de borde
        self.grid[1:-1, 1:-1] = interior.T

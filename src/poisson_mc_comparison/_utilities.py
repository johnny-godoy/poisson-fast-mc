"""Implement utility functions for all MC solvers."""
from __future__ import annotations

from random import random


def get_random_walk(i_walk: int, j_walk: int, n_steps: int) -> tuple[int, int,
                                                                     list[tuple[int, int]]]:
    """Simulate a random walk with n_steps steps starting at (i_walk, j_walk).
    Parameters
    ----------
    i_walk: int
        The initial row of the walk.
    j_walk: int
        The initial column of the walk.
    n_steps: int
        The number of steps of the walk.
    Returns
    -------
    i_walk: int
        The final row of the walk.
    j_walk: int
        The final column of the walk.
    walk: list[tuple[int, int]]
        A list of the interior steps of the walk."""
    walk = []
    while True:
        walk.append((i_walk, j_walk))
        rand = random()
        if rand <= .25:
            i_walk += 1
            if i_walk == n_steps:
                break
        elif rand <= .5:
            i_walk -= 1
            if i_walk == 0:
                break
        elif rand <= .75:
            j_walk += 1
            if j_walk == n_steps:
                break
        else:
            j_walk -= 1
            if j_walk == 0:
                break
    return i_walk, j_walk, walk

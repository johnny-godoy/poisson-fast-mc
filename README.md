# Monte Carlo solver for the Poisson Equation

In this repository, we provide a Python library with solvers for the [Poisson equation](https://en.wikipedia.org/wiki/Poisson_equation) on a square domain with Dirichlet conditions using Monte Carlo methods, including our own custom solver with significant optimizations.

# Installation and usage
To install the Python library, run the following command:

```
pip install git+https://github.com/johnny-godoy/poisson-fast-mc.git
```
After that, you can import solvers from the following files:

* ```monte_carlo_solvers.py``` Uses the usual implementation, obtained from [here](https://github.com/s-ankur/montecarlo-pde). It also adds bayesian estimations for the probability of an endpoint being reached from a starting point.
* ```subwalk_monte_carlo_solvers.py``` Uses our own custom implementation which improves the runtime by reusing every subwalk of the generated random walks.
* ```finite_difference_solver.py``` The usual finite difference solver.
* ```hybrid_solver.py``` A hybrid solver that uses the subwalk solver to obtain a primer, and then uses conjugate gradient to solve the finite difference system.

With the Python code:

```
from poisson_mc_comparison import {solver_filename}
```

# Reports
The ``report`` directory contains a Jupyter Notebook, a poster and a slideshow,
and were designed to explain the algorithms and their performances. Note that these are in Spanish. You may also view the Jupyter Notebook report as HTML [here](https://johnny-godoy.github.io/poisson-fast-mc/).

# About

The reports were done as a capstone project for the course MA5307: "Numerical Analysis in Partial Differential Equations, Theory and Practice",
taught by Axel Osses at Universidad de Chile, Department of Mathematical Engineering.

The original subject of this project was to compare the performance of a simple MC solver with the finite difference solver,
but we ended up also designing the subwalk solver to improve performance over the usual MC solver.

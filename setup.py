"""Setup installation file."""

import setuptools

setuptools.setup(name="poisson_mc_comparison",
                 version="1.0.0",
                 description="Fast Monte Carlo solver for the Poisson problem",
                 license="MIT",
                 author="Johnny Godoy, Javier Santidrián, Patricio Yáñez",
                 author_email="johnny.godoy@ing.uchile.cl",
                 install_requires=["numpy>=1.21.6", "scipy>=1.7.3"],
                 packages=setuptools.find_packages("src"),
                 package_dir={"": "src"},
                 )

.. _installation:

===========================
Installation
===========================

Requirements
------------

Fluopy requires Python 3.12 or newer. Installation with pip also installs the
required scientific Python libraries. The complete dependency list is maintained
in pyproject.toml in the repository.

Install from PyPI
------------------------------

Install Fluopy directly from the Python Package Index::

    python -m pip install fluopy

Install the development version
------------------------------

To install the latest changes from the GitHub main branch::

    python -m pip install git+https://github.com/super-resolution/Fluopy.git@main

Jupyter notebooks
-----------------

To run the :doc:`tutorial notebooks <../tutorials/tutorials>` interactively,
install JupyterLab in the same environment as Fluopy::

    python -m pip install jupyterlab
    jupyter lab

Working from a source checkout
------------------------------

For development, clone the repository and use uv to install Fluopy together with
the test and notebook dependency groups::

    git clone https://github.com/super-resolution/Fluopy.git
    cd Fluopy
    uv sync --group test --group jupyter

Run tests or open the notebooks from the repository root::

    uv run --no-sync python -m pytest
    uv run --no-sync jupyter lab

The dependency groups are defined in the repository's pyproject.toml; they are
not extras supplied by an installed Fluopy package.

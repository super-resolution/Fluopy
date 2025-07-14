.. _installation:

===========================
Installation
===========================

Dependencies
------------

* python 3
* standard scipy and other open source libraries

A list with all hard and optional dependencies is given in `pyproject.toml`.

Install from pypi
------------------------------

Install locan directly from the Python Package Index::

    pip install fluopy

Extra dependencies can be included::

    pip install fluopy[gpu]

Install from distribution or sources
-------------------------------------

In order to get the latest changes install from the GitHub repository
main branch::

    pip install git+https://github.com/super-resolution/Photoswitching.git@main

or download distribution or wheel archive and install with pip::

    pip install <distribution_file>

Run tests
-----------------------

Use pytest to run tests from the source or tests directory::

    pip install --group test
    pytest


Jupyter
-----------------------

To work with jupyter notebooks install dependencies from the dependency-group
jupyter::

    pip install --group jupyter

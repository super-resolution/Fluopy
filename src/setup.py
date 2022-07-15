from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(ext_modules=cythonize("ssa_cython.pyx"),
      include_dirs=[numpy.get_include()],
      zip_safe=False,
      package_dir={"src": ""},
      name="cython ssa algorithm")

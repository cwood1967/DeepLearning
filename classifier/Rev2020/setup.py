from setuptools import setup
from Cython.Build import cythonize

setup(
    name="lookup",
    ext_modules=cythonize("lookup_cy.pyx"),
    zip_safe=False,
)
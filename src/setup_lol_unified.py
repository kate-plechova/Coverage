from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent

extensions = [
    Extension(
        "src._lol_unified_core",
        sources=[str(HERE / "_lol_unified_core.pyx")],
        language="c++",
        include_dirs=[np.get_include(), str(HERE)],
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "src._lol5_core",
        sources=[str(HERE / "_lol5_core.pyx")],
        language="c++",
        include_dirs=[np.get_include(), str(HERE)],
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="coverage_cores",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
)
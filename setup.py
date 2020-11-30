#!/usr/bin/env python
import os
import re
import sys
import warnings
import versioneer
from setuptools import setup, find_packages

DISTNAME = "fastjmd95"
LICENSE = "MIT"
AUTHOR = "fastjmd95 Developers"
AUTHOR_EMAIL = "rpa@ldeo.columbia.edu"
URL = "https://github.com/xgcm/fastjmd95"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = ["numba"]
EXTRAS_REQUIRE = {"dask": ["dask"], "xarray": ["xarray"]}
PYTHON_REQUIRES = ">=3.6"

DESCRIPTION = "Numba version of Jackett & McDougall (1995) ocean equation of state."


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=readme(),
    install_requires=INSTALL_REQUIRES,
    python_requires=PYTHON_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    url=URL,
    packages=find_packages(),
)

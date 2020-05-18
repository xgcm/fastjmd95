fastjmd95: Numba implementation of Jackett & McDougall (1995) ocean equation of state
=====================================================================================

|Build Status| |license|

This package provides a Numba_ implementation the JMD95 equation of state.

Usage
-----

fastjmd95 provides three ufuncs:

.. code-block:: python

   >>> from fastjmd95 import rho, drhods, drhodt
   >>> rho(35.5, 3., 3000.)
   1041.83267
   >>> drhodt(35.5, 3., 3000.)
   -0.17244
   >>> drhods(35.5, 3., 3000.)
   0.77481

Tutorial
--------

Tutorial notebook located at `doc/fastjmd95_tutorial.ipynb <https://nbviewer.jupyter.org/github/xgcm/fastjmd95/blob/master/doc/fastjmd95_tutorial.ipynb>`_.

.. _Pangeo: http://pangeo-data.github.io
.. _Numba: http://numba.pydata.org/

.. |conda forge| image:: https://anaconda.org/conda-forge/fastjmd95/badges/version.svg
   :target: https://anaconda.org/conda-forge/fastjmd95
.. |Build Status| image:: https://travis-ci.org/xgcm/fastjmd95.svg?branch=master
   :target: https://travis-ci.org/xgcm/fastjmd95
   :alt: travis-ci build status
.. |codecov| image:: https://codecov.io/github/xgcm/fastjmd95/coverage.svg?branch=master
   :target: https://codecov.io/github/xgcm/fastjmd95?branch=master
   :alt: code coverage
.. |pypi| image:: https://badge.fury.io/py/fastjmd95.svg
   :target: https://badge.fury.io/py/fastjmd95
   :alt: pypi package
.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/xgcm/fastjmd95
   :alt: license

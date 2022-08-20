.. nndcp documentation master file, created by
   sphinx-quickstart on Sun Apr 25 19:17:27 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NNDCP 0.0.1!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

**Training neural network with difference of convex programming (DCP)**

This is the API documentation for Python package ``nndcp``, designed for training neural network with DC algorithm.
This pakcage have four different modules, aiming for different tasks.
For more details, please refer to the GitHub page for this project (here_).
You will also find some examples for using this package there.

.. _here: https://github.umn.edu/liu00980/nndcp


API
===

Module ``DCshallow``
--------------------
.. automodule:: DCshallow
   :members:

Module ``SGDtraining``
----------------------
.. automodule:: SGDtraining
   :members:

Module ``data``
---------------
.. automodule:: data
   :members: 

Module ``utils.util``
---------------------
.. automodule:: utils.util
   :members: relunn, eloss, f, fa, fb, g, ga, gb, normal_nn, todataset, extract, splitdataset, wholeloss, 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

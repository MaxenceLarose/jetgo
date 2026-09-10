jetgo
=====

*Jet Event Toolkit for Generator-level Observables.*

``jetgo`` generates collider events with `pythia8mc <https://pypi.org/project/pythia8mc/>`_,
clusters the particles into jets with `fastjet <https://pypi.org/project/fastjet/>`_, tags each jet
with the color charge of the parton that initiated it, and writes the observables out. All you define
is the simulator, the observables and the tagger.

.. code-block:: bash

   pip install jetgo[pythia]

PYTHIA8 is an optional dependency, because the ``pythia8mc`` wheels are Linux-only. Plain
``pip install jetgo`` works everywhere and gives you every observable and tagger; only event
generation needs the extra.

There are two ways to get PYTHIA8, and ``jetgo`` accepts either:

**The wheel.**
   The ``[pythia]`` extra above pulls in `pythia8mc <https://pypi.org/project/pythia8mc/>`_, the
   PyPI distribution of PYTHIA's own Python bindings. Nothing to compile, but the wheels are built
   for Linux only.

**Your own build.**
   Compile PYTHIA8 with its Python interface enabled, following `the PYTHIA manual
   <https://pythia.org/latest-manual/PythonInterface.html>`_, so that ``import pythia8`` works.
   This is the route on macOS and Windows, and the one to take if you need a specific PYTHIA
   version or your own patches.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   quickstart
   custom_observable
   custom_tagger

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`

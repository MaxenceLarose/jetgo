jetgo
=====

*Jet Event Toolkit for Generating Observables.*

``jetgo`` generates collider events, clusters the particles into jets, tags each jet with the
color charge of the parton that initiated it, and writes the observables out.

You define the simulation, the observables and the tagger. ``jetgo`` runs the loop.

.. code-block:: bash

   pip install jetgo[pythia]

PYTHIA8 is an optional dependency, because the ``pythia8mc`` wheels are Linux-only. Plain
``pip install jetgo`` works everywhere and gives you every observable and tagger; only event
generation needs the extra.

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

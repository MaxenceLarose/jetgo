Quickstart
==========

A run is three collaborating objects handed to a :class:`~jetgo.simulation.simulator.Simulator`,
plus the observables you want out of it.

The three pieces
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Responsibility
   * - :class:`~jetgo.simulation.event_generator.EventGenerator`
     - Configures PYTHIA8 for hard-QCD collisions in a p̂T window, and reports the cross-section
       normalization.
   * - :class:`~jetgo.simulation.particle_selector.ParticleSelector`
     - Keeps final-state, visible, optionally charged particles inside an acceptance, and converts
       them to FastJet inputs.
   * - :class:`~jetgo.simulation.jet_finder.JetFinder`
     - Holds a jet definition and the kinematic cuts a jet must pass.

A first run
-----------

.. code-block:: python

   import numpy as np

   from jetgo.observables import EEC
   from jetgo.simulation import EventGenerator, JetFinder, ParticleSelector, Simulator

   simulator = Simulator(
       event_generator=EventGenerator(
           beam_1=2212, beam_2=2212, sqrt_s=5020, pt_min=80, pt_max=200, random_seed=42
       ),
       jet_finder=JetFinder(radius=0.4, pt_min=120, pt_max=140, max_abs_pseudorapidity=1.6),
       particle_selector=ParticleSelector(max_abs_pseudorapidity=2.4),
   )

   simulator.simulate(
       n_events=100_000,
       observables=EEC(bin_edges=np.geomspace(0.005, 0.8, 31), charged_only=True),
       output_path="jet_eec.json",
   )

.. warning::

   ``random_seed=0`` lets PYTHIA8 choose its own seed from the clock, and the run stops being
   reproducible. Pass an explicit non-zero seed for anything you intend to repeat.

Why the bin edges are required
------------------------------

Observables histogram at fill time rather than storing each contribution. An energy-energy
correlator holds one entry per constituent *pair*, which grows quadratically with jet
multiplicity, so a million-event run would cost gigabytes of memory and JSON to keep values that
are only ever histogrammed onto one known binning anyway. Binning as you fill makes the cost
independent of run length.

The consequence is that the edges are not the observable's to choose. They come from whichever
measurement or analysis the run will be read against, so every observable takes them as a
required argument.

Splitting by flavor
-------------------

Pass one or more taggers and every observable is filled three times over: once with all jets, and
once each with the jets a tagger calls quark- or gluon-initiated.

.. code-block:: python

   from jetgo.taggers import DescendancyTracing

   simulator.simulate(
       n_events=100_000,
       observables=EEC(bin_edges=np.geomspace(0.005, 0.8, 31), charged_only=True),
       jet_flavor_taggers=DescendancyTracing(delta_r_max=0.4),
       output_path="jet_eec.json",
   )

:class:`~jetgo.taggers.descendancy_tracing.DescendancyTracing` never leaves a jet untagged. A
tagger you write yourself might, and those jets are excluded from both flavor histograms while
still counting towards the inclusive one.

What the output file holds
--------------------------

One JSON file per run::

   {
     "metadata": {
       "n_events": ...,
       "n_jets": ...,
       "flavor_taggers": {"<tagger>": {"n_quark": ..., "n_gluon": ..., "n_untagged": ...}},
       "pairwise_flavor_agreement": [...]
     },
     "<observable identifier>": {
       "histograms": {
         "inclusive": {"values": [...], "metadata": {...}},
         "<tagger>": {"quark": {...}, "gluon": {...}}
       }
     }
   }

Reading it back
---------------

The saved ``values`` are raw accumulated bins. Each observable's ``finalize`` classmethod turns
them into the normalized quantity, and it runs downstream, never during the simulation.

.. code-block:: python

   import json

   result = json.loads(open("jet_eec.json").read())
   saved = result["Energy-energy correlator"]["histograms"]["inclusive"]

   shape = EEC.finalize(saved["values"], np.geomspace(0.005, 0.8, 31), saved["metadata"])

``finalize`` will not silently re-bin. The edges you pass are you stating which binning you
believe you are reading, and each observable either serves it or raises.

Writing your own observable
===========================

Observables are the main extension point. Subclass
:class:`~jetgo.observables.base.Observable` and implement six methods.

The contract
------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Method
     - What it does
   * - ``identifier``
     - This observable's key in the output file. Must be unique within a run.
   * - ``fill(jets, cluster)``
     - Called once per event with that event's jets. Add their contributions to your histogram.
   * - ``scale(scale_factor)``
     - Record the generator's cross-section normalization, applied later. A no-op for
       self-normalized shapes.
   * - ``finalize(values, bin_edges, metadata)``
     - A classmethod, run downstream, turning the saved histogram into the normalized quantity.
   * - ``_get_complementary_metadata()``
     - Whatever ``finalize`` will need, including the bin edges you filled on.
   * - ``_get_binned_values()``
     - The accumulated histogram, one value per bin.

Two rules worth internalizing
-----------------------------

**Bin in fill, not in finalize.** The edges are fixed when the observable is constructed. This is
what keeps a million-event run's output small, and it is why every shipped observable takes
``bin_edges`` as a required argument.

**Never silently re-bin.** ``finalize`` receives the edges the caller believes it is reading. If
you cannot serve exactly those, raise. :class:`~jetgo.observables.eec.EEC` refuses anything but
the edges it was filled on. :class:`~jetgo.observables.double_differential_jet_cross_section.DoubleDifferentialJetCrossSection`
sums whole bins onto coarser edges and raises if a requested edge falls between two of its own.
Returning values on edges other than the ones asked for is the one thing an implementation must
not do.

A worked example
----------------

Jet constituent multiplicity, a quantity that separates quark and gluon jets:

.. code-block:: python

   from typing import Dict, List

   import fastjet as fj
   import numpy as np

   from jetgo.observables.base import Observable


   class ConstituentMultiplicity(Observable):
       N_JETS_KEY = "n_jets"
       BIN_EDGES_KEY = "bin_edges"

       def __init__(self, bin_edges: np.ndarray, pt_min: float = 1.0) -> None:
           self._bin_edges = np.asarray(bin_edges, dtype=float)
           self._counts = np.zeros(len(self._bin_edges) - 1, dtype=float)
           self._pt_min = pt_min
           self._n_jets = 0

       @property
       def identifier(self) -> str:
           return "constituent_multiplicity"

       def fill(self, jets: List[fj.PseudoJet], cluster: fj.ClusterSequence) -> None:
           for jet in jets:
               self._n_jets += 1
               multiplicity = sum(1 for c in jet.constituents() if c.pt() > self._pt_min)

               index = int(np.searchsorted(self._bin_edges, multiplicity, side="right")) - 1
               if 0 <= index < len(self._counts):
                   self._counts[index] += 1.0

       def scale(self, scale_factor: float) -> None:
           pass  # self-normalized: the cross-section factor cancels in the shape

       @classmethod
       def finalize(cls, values, bin_edges: np.ndarray, metadata: Dict) -> np.ndarray:
           counts = np.asarray(values, dtype=float)
           return counts / counts.sum() / np.diff(np.asarray(bin_edges, dtype=float))

       def _get_complementary_metadata(self) -> Dict:
           return {self.N_JETS_KEY: self._n_jets, self.BIN_EDGES_KEY: self._bin_edges.tolist()}

       def _get_binned_values(self) -> List[float]:
           return self._counts.tolist()

Hand it to ``simulate`` exactly like a shipped observable. When taggers are attached, the
simulator deep-copies your observable into an independent, empty quark and gluon pair per tagger,
so nothing in your ``__init__`` needs to know that flavor splitting exists.

Reading a particle's identity
-----------------------------

FastJet's ``PseudoJet`` carries no particle identity, only a four-momentum and a single integer
slot. ``jetgo`` uses that slot to carry the particle's Pythia8 event-record index and its charge,
as magnitude and sign. Read it through :mod:`jetgo.kinematics` rather than touching
``user_index()`` yourself:

.. code-block:: python

   from jetgo.kinematics import delta_r, is_charged, pythia_index

   for constituent in jet.constituents():
       if is_charged(constituent):
           ...

The full example is in ``examples/ex03_custom_observable.py``.

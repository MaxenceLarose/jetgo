Writing your own tagger
=======================

A tagger answers one question about one jet: was it initiated by a quark or a gluon? Subclass
:class:`~jetgo.taggers.base.JetFlavorTagger` and implement ``identifier`` and ``tag``.

.. code-block:: python

   import fastjet as fj

   from jetgo.taggers.base import JetFlavorTagger
   from jetgo.taggers.flavor import JetFlavor


   class MyTagger(JetFlavorTagger):

       @property
       def identifier(self) -> str:
           return "my_tagger"

       def tag(self, jet: fj.PseudoJet, event) -> JetFlavor:
           ...
           return JetFlavor.QUARK  # or GLUON, or UNTAGGED

The ``event`` handed to ``tag`` is the untouched PYTHIA8 record for the event the jet was
reconstructed from, so the full particle history is available. Return ``JetFlavor.UNTAGGED``
rather than guessing when the answer is not there: untagged jets are excluded from the flavor
histograms but still counted inclusively, so the inclusive result stays correct.

The base class gives you ``_get_flavor_from_pdg_id``, which maps a PDG id onto a flavor, and
``get_diagnostics``, which you can override to fold your own counters into the run metadata.

Comparing taggers
-----------------

Pass several taggers to one run and each gets its own quark and gluon histograms. The output
metadata then also reports, for every pair, how often they agreed on the jets both of them
tagged:

.. code-block:: python

   simulator.simulate(
       ...,
       jet_flavor_taggers=[DescendancyTracing(delta_r_max=0.4), MyTagger()],
   )

Every jet is tagged exactly once per tagger in a single pass, so the comparison is on identical
jets rather than on two separate runs. This is the check worth running on any new definition:
two strategies recovering the same physical truth should mostly agree, and where they do not is
where the definition is doing real work.

The shipped tagger
------------------

:class:`~jetgo.taggers.descendancy_tracing.DescendancyTracing` starts from the outgoing hard
partons, those with PYTHIA status 23, and walks their shower descendants *forward*. Descendants
that land within ``delta_r_max`` of the jet axis vote for their ancestor's flavor, weighted by
transverse momentum.

The direction matters. Tracing a constituent's mother list *backward* is the more obvious
construction, but color reconnection rewrites those mother links, so a hadron's recorded ancestry
need not be the shower it physically came from. Descendancy tracing reads only
``daughterListRecursive`` from the hard partons outward, which color reconnection does not touch.

Jets whose lineage traces to the beam remnant rather than to the hard scattering have no
status-23 ancestor to vote for them. With ``handle_untagged_jets=True``, the default, an explicit
fallback finds the relevant beam-remnant partons instead, so no jet is discarded, and the number
of times it fired is reported in the run metadata as ``fallback_call_count``.

That path is rare. Across 7,999 jets above 100 GeV in a 5.02 TeV proton-proton run at R = 0.4, the
fallback fired for 0.3% of them and left nothing untagged; with ``handle_untagged_jets=False``,
those same 0.3% come back as ``JetFlavor.UNTAGGED`` instead. Pass ``False`` if you would rather
see them marked untagged than resolved against the beam remnant.

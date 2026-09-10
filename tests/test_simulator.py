"""
    @file:              test_simulator.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       Tests for the simulation loop, driven end to end with a stand-in generator, particle
                        selector and taggers, so that the whole pipeline including the output file is covered
                        without Pythia8. See ``conftest.py`` for the stand-ins.
"""

import json

import numpy as np
import pytest

from jetgo.observables import DoubleDifferentialJetCrossSection, EEC
from jetgo.observables.base import Observable
from jetgo.simulation import JetFinder, Simulator
from jetgo.taggers.flavor import JetFlavor

from conftest import StubEventGenerator, StubParticleSelector, StubTagger

QUARK, GLUON, UNTAGGED = JetFlavor.QUARK, JetFlavor.GLUON, JetFlavor.UNTAGGED


@pytest.fixture
def simulator(two_prong_event):
    return Simulator(
        event_generator=StubEventGenerator(n_events=10, scale_factor=3.0),
        jet_finder=JetFinder(radius=0.4),
        particle_selector=StubParticleSelector(two_prong_event),
    )


def cross_section() -> DoubleDifferentialJetCrossSection:
    return DoubleDifferentialJetCrossSection(
        bin_edges=np.linspace(0.0, 200.0, 21), max_abs_rapidity=1.0
    )


def run(simulator, tmp_path, observables, taggers=None) -> dict:
    output = tmp_path / "run.json"
    simulator.simulate(
        n_events=10, observables=observables, output_path=output, jet_flavor_taggers=taggers
    )
    return json.loads(output.read_text())


class TestOutputStructure:

    def test_writes_metadata_and_one_entry_per_observable(self, simulator, tmp_path):
        result = run(simulator, tmp_path, [cross_section(), EEC(bin_edges=np.geomspace(0.01, 0.5, 6))])

        assert set(result) == {"metadata", "D2SIG/DPT/DYRAP", "Energy-energy correlator"}

    def test_counts_every_event_and_jet(self, simulator, tmp_path):
        result = run(simulator, tmp_path, cross_section())

        assert result["metadata"]["n_events"] == 10
        # Two jets per event, from the two-prong stand-in event.
        assert result["metadata"]["n_jets"] == 20

    def test_a_single_observable_need_not_be_wrapped_in_a_sequence(self, simulator, tmp_path):
        result = run(simulator, tmp_path, cross_section())

        assert "D2SIG/DPT/DYRAP" in result

    def test_without_taggers_only_the_inclusive_histogram_is_written(self, simulator, tmp_path):
        result = run(simulator, tmp_path, cross_section())

        assert set(result["D2SIG/DPT/DYRAP"]["histograms"]) == {"inclusive"}
        assert result["metadata"]["flavor_taggers"] == {}

    def test_the_scale_factor_reaches_the_saved_metadata(self, simulator, tmp_path):
        result = run(simulator, tmp_path, cross_section())
        metadata = result["D2SIG/DPT/DYRAP"]["histograms"]["inclusive"][Observable.METADATA_KEY]

        assert metadata[DoubleDifferentialJetCrossSection.SCALE_FACTOR_KEY] == 3.0


class TestFlavorSplitting:

    def test_each_tagger_gets_its_own_quark_and_gluon_histogram(self, simulator, tmp_path):
        tagger = StubTagger("stub", [QUARK, GLUON])

        result = run(simulator, tmp_path, cross_section(), taggers=tagger)

        assert set(result["D2SIG/DPT/DYRAP"]["histograms"]) == {"inclusive", "stub"}
        assert set(result["D2SIG/DPT/DYRAP"]["histograms"]["stub"]) == {"quark", "gluon"}

    def test_quark_and_gluon_counts_add_up_to_the_jets_the_tagger_labelled(self, simulator, tmp_path):
        result = run(simulator, tmp_path, cross_section(), taggers=StubTagger("stub", [QUARK, GLUON]))

        counts = result["metadata"]["flavor_taggers"]["stub"]
        assert counts["n_quark"] == 10
        assert counts["n_gluon"] == 10
        assert counts["n_untagged"] == 0

    def test_untagged_jets_are_excluded_from_both_flavors_but_kept_inclusively(self, simulator, tmp_path):
        result = run(simulator, tmp_path, cross_section(), taggers=StubTagger("stub", [QUARK, UNTAGGED]))

        histograms = result["D2SIG/DPT/DYRAP"]["histograms"]
        counts = result["metadata"]["flavor_taggers"]["stub"]

        assert counts["n_quark"] == 10 and counts["n_untagged"] == 10
        assert sum(histograms["inclusive"][Observable.VALUES_KEY]) == 20
        assert sum(histograms["stub"]["quark"][Observable.VALUES_KEY]) == 10
        assert sum(histograms["stub"]["gluon"][Observable.VALUES_KEY]) == 0

    def test_flavor_histograms_are_independent_copies_of_the_inclusive_one(self, simulator, tmp_path):
        """Each copy must start empty rather than inherit the inclusive observable's contents."""
        observable = cross_section()
        observable.fill(jets=[], cluster=None)

        result = run(simulator, tmp_path, observable, taggers=StubTagger("stub", [QUARK]))
        histograms = result["D2SIG/DPT/DYRAP"]["histograms"]

        assert sum(histograms["stub"]["quark"][Observable.VALUES_KEY]) == 20
        assert sum(histograms["stub"]["gluon"][Observable.VALUES_KEY]) == 0

    def test_tagger_diagnostics_are_folded_into_the_metadata(self, simulator, tmp_path):
        tagger = StubTagger("stub", [QUARK], diagnostics={"fallback_call_count": 7})

        result = run(simulator, tmp_path, cross_section(), taggers=tagger)

        assert result["metadata"]["flavor_taggers"]["stub"]["fallback_call_count"] == 7


class TestPairwiseAgreement:

    def _agreement(self, simulator, tmp_path, flavors_a, flavors_b):
        result = run(
            simulator, tmp_path, cross_section(),
            taggers=[StubTagger("a", flavors_a), StubTagger("b", flavors_b)],
        )
        entries = result["metadata"]["pairwise_flavor_agreement"]
        assert len(entries) == 1
        return entries[0]

    def test_two_taggers_that_always_agree_report_full_agreement(self, simulator, tmp_path):
        entry = self._agreement(simulator, tmp_path, [QUARK, GLUON], [QUARK, GLUON])

        assert entry["taggers"] == ["a", "b"]
        assert entry["n_both_tagged"] == 20
        assert entry["n_agree"] == 20
        assert entry["agreement_fraction"] == pytest.approx(1.0)

    def test_two_taggers_that_never_agree_report_none(self, simulator, tmp_path):
        entry = self._agreement(simulator, tmp_path, [QUARK], [GLUON])

        assert entry["n_both_tagged"] == 20
        assert entry["n_agree"] == 0
        assert entry["agreement_fraction"] == pytest.approx(0.0)

    def test_jets_only_one_tagger_labels_are_excluded_from_the_comparison(self, simulator, tmp_path):
        entry = self._agreement(simulator, tmp_path, [QUARK, QUARK], [QUARK, UNTAGGED])

        assert entry["n_both_tagged"] == 10
        assert entry["n_agree"] == 10

    def test_a_pair_that_never_both_tag_reports_no_fraction(self, simulator, tmp_path):
        entry = self._agreement(simulator, tmp_path, [UNTAGGED], [QUARK])

        assert entry["n_both_tagged"] == 0
        assert entry["agreement_fraction"] is None

    def test_three_taggers_give_three_pairs(self, simulator, tmp_path):
        result = run(
            simulator, tmp_path, cross_section(),
            taggers=[StubTagger(name, [QUARK]) for name in ("a", "b", "c")],
        )

        assert len(result["metadata"]["pairwise_flavor_agreement"]) == 3


class TestIdentifierUniqueness:

    def test_duplicate_observable_identifiers_are_refused(self, simulator, tmp_path):
        with pytest.raises(ValueError, match="observable"):
            run(simulator, tmp_path, [cross_section(), cross_section()])

    def test_duplicate_tagger_identifiers_are_refused(self, simulator, tmp_path):
        with pytest.raises(ValueError, match="tagger"):
            run(simulator, tmp_path, cross_section(),
                taggers=[StubTagger("same", [QUARK]), StubTagger("same", [GLUON])])

    def test_distinct_identifiers_are_accepted(self, simulator, tmp_path):
        result = run(simulator, tmp_path, [cross_section(), EEC(bin_edges=np.geomspace(0.01, 0.5, 6))],
                     taggers=[StubTagger("a", [QUARK]), StubTagger("b", [GLUON])])

        assert set(result["metadata"]["flavor_taggers"]) == {"a", "b"}


class TestRoundTrip:

    def test_the_saved_histogram_finalizes_back_to_the_expected_cross_section(self, simulator, tmp_path):
        edges = np.linspace(0.0, 200.0, 21)
        result = run(simulator, tmp_path, cross_section())
        saved = result["D2SIG/DPT/DYRAP"]["histograms"]["inclusive"]

        finalized = DoubleDifferentialJetCrossSection.finalize(
            saved[Observable.VALUES_KEY], np.array([0.0, 200.0]), saved[Observable.METADATA_KEY]
        )

        # 20 jets * 3.0 scale / (200 GeV * 2.0 rapidity interval)
        assert finalized.tolist() == pytest.approx([0.15])
        assert len(saved[Observable.VALUES_KEY]) == len(edges) - 1

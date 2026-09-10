"""
    @file:              test_observables.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       Tests for the shipped observables, covering the binning contract each one enforces at
                        ``finalize`` time and the normalization it applies.
"""

import numpy as np
import pytest

from jetgo.observables import DoubleDifferentialJetCrossSection, EEC
from jetgo.observables.base import Observable
from jetgo.observables.classes import OBSERVABLE_CLASSES
from jetgo.observables.identifier import ObservableIdentifier

from conftest import pseudo_jet


class TestObservableRegistry:

    def test_every_identifier_maps_to_an_observable_subclass(self):
        assert set(OBSERVABLE_CLASSES) == set(ObservableIdentifier)

        for identifier, cls in OBSERVABLE_CLASSES.items():
            assert issubclass(cls, Observable)
            assert cls(bin_edges=np.linspace(1.0, 2.0, 3), **(
                {"max_abs_rapidity": 1.0} if identifier is ObservableIdentifier.D2SIG_DPT_DYRAP else {}
            )).identifier is identifier


class TestDoubleDifferentialJetCrossSectionAcceptance:

    def test_rejects_both_acceptance_arguments(self):
        with pytest.raises(ValueError, match="Exactly one of"):
            DoubleDifferentialJetCrossSection(
                bin_edges=np.linspace(100.0, 200.0, 3), max_abs_rapidity=1.0, max_abs_pseudorapidity=1.0
            )

    def test_rejects_neither_acceptance_argument(self):
        with pytest.raises(ValueError, match="Exactly one of"):
            DoubleDifferentialJetCrossSection(bin_edges=np.linspace(100.0, 200.0, 3))

    @pytest.mark.parametrize("kwargs", [{"max_abs_rapidity": 1.5}, {"max_abs_pseudorapidity": 1.5}])
    def test_either_acceptance_alone_sets_the_interval_width(self, kwargs):
        observable = DoubleDifferentialJetCrossSection(bin_edges=np.linspace(100.0, 200.0, 3), **kwargs)
        metadata = observable.to_dict()[Observable.METADATA_KEY]

        assert metadata[DoubleDifferentialJetCrossSection.RAPIDITY_INTERVAL_KEY] == pytest.approx(3.0)


class TestDoubleDifferentialJetCrossSectionFill:

    def test_counts_land_in_the_right_bin(self):
        observable = DoubleDifferentialJetCrossSection(
            bin_edges=np.array([100.0, 110.0, 120.0]), max_abs_rapidity=1.0
        )
        observable.fill(jets=[pseudo_jet(pt=105.0), pseudo_jet(pt=115.0), pseudo_jet(pt=118.0)], cluster=None)

        assert observable.to_dict()[Observable.VALUES_KEY] == [1.0, 2.0]

    def test_jets_outside_the_grid_are_dropped(self):
        observable = DoubleDifferentialJetCrossSection(
            bin_edges=np.array([100.0, 110.0]), max_abs_rapidity=1.0
        )
        observable.fill(jets=[pseudo_jet(pt=50.0), pseudo_jet(pt=105.0), pseudo_jet(pt=500.0)], cluster=None)

        assert observable.to_dict()[Observable.VALUES_KEY] == [1.0]


class TestDoubleDifferentialJetCrossSectionRebinning:

    FILLED_EDGES = np.array([100.0, 110.0, 120.0, 130.0, 140.0])

    def test_sums_whole_bins_onto_coarser_edges(self):
        counts = np.array([1.0, 2.0, 3.0, 4.0])

        rebinned = DoubleDifferentialJetCrossSection._rebin(
            counts, self.FILLED_EDGES, np.array([100.0, 120.0, 140.0])
        )

        assert rebinned.tolist() == [3.0, 7.0]

    def test_the_identity_rebinning_returns_the_counts_unchanged(self):
        counts = np.array([1.0, 2.0, 3.0, 4.0])

        rebinned = DoubleDifferentialJetCrossSection._rebin(counts, self.FILLED_EDGES, self.FILLED_EDGES)

        assert rebinned.tolist() == counts.tolist()

    def test_rejects_an_edge_that_falls_between_grid_edges(self):
        """The individual jet pT values are gone, so a partial bin cannot be reconstructed."""
        with pytest.raises(ValueError, match="does not lie on the grid"):
            DoubleDifferentialJetCrossSection._rebin(
                np.array([1.0, 2.0, 3.0, 4.0]), self.FILLED_EDGES, np.array([100.0, 115.0, 140.0])
            )

    def test_finalize_normalizes_by_bin_width_and_rapidity_interval(self):
        observable = DoubleDifferentialJetCrossSection(bin_edges=self.FILLED_EDGES, max_abs_rapidity=1.0)
        observable.fill(jets=[pseudo_jet(pt=105.0), pseudo_jet(pt=115.0)], cluster=None)
        observable.scale(4.0)

        saved = observable.to_dict()
        finalized = DoubleDifferentialJetCrossSection.finalize(
            saved[Observable.VALUES_KEY], np.array([100.0, 120.0]), saved[Observable.METADATA_KEY]
        )

        # 2 jets * 4.0 scale / (20 GeV bin width * 2.0 rapidity interval)
        assert finalized.tolist() == pytest.approx([0.2])

    def test_finalize_refuses_a_file_with_no_saved_grid(self):
        with pytest.raises(ValueError, match="individual jet pT values"):
            DoubleDifferentialJetCrossSection.finalize(
                [1.0], np.array([100.0, 110.0]),
                {DoubleDifferentialJetCrossSection.RAPIDITY_INTERVAL_KEY: 2.0,
                 DoubleDifferentialJetCrossSection.SCALE_FACTOR_KEY: 1.0},
            )


class TestEECBinningContract:

    BIN_EDGES = np.geomspace(0.01, 0.5, 11)

    def _filled_eec(self, **kwargs) -> EEC:
        observable = EEC(bin_edges=self.BIN_EDGES, pt_min=0.0, **kwargs)
        jet = pseudo_jet(pt=100.0)
        constituents = [
            pseudo_jet(pt=50.0, eta=0.0, phi=0.0, pythia_event_index=1, charged=True),
            pseudo_jet(pt=50.0, eta=0.1, phi=0.0, pythia_event_index=2, charged=True),
        ]

        class _Jet:
            def pt(self):
                return jet.pt()

            def constituents(self):
                return constituents

        observable.fill(jets=[_Jet()], cluster=None)
        return observable

    def test_finalize_serves_the_edges_it_was_filled_on(self):
        observable = self._filled_eec()
        saved = observable.to_dict()

        finalized = EEC.finalize(saved[Observable.VALUES_KEY], self.BIN_EDGES, saved[Observable.METADATA_KEY])

        assert len(finalized) == len(self.BIN_EDGES) - 1
        # Unity normalized: the integral over ΔR is one.
        assert float(np.sum(finalized * np.diff(self.BIN_EDGES))) == pytest.approx(1.0)

    def test_finalize_refuses_different_edges(self):
        """Unlike the cross-section, the EEC will not re-bin: the pairs are gone."""
        observable = self._filled_eec()
        saved = observable.to_dict()

        with pytest.raises(ValueError, match="was filled on"):
            EEC.finalize(saved[Observable.VALUES_KEY], np.geomspace(0.01, 0.5, 6),
                         saved[Observable.METADATA_KEY])

    def test_finalize_refuses_the_same_range_with_a_different_count(self):
        observable = self._filled_eec()
        saved = observable.to_dict()

        with pytest.raises(ValueError, match="was filled on"):
            EEC.finalize(saved[Observable.VALUES_KEY], np.geomspace(0.01, 0.5, 12),
                         saved[Observable.METADATA_KEY])

    def test_finalize_refuses_a_file_with_no_saved_edges(self):
        with pytest.raises(ValueError, match="raw"):
            EEC.finalize([1.0], self.BIN_EDGES, {EEC.W_PAIRS_KEY: 1.0, EEC.N_JETS_KEY: 1})

    def test_scale_is_a_no_op_because_the_eec_is_self_normalized(self):
        observable = self._filled_eec()
        before = list(observable.to_dict()[Observable.VALUES_KEY])

        observable.scale(1234.5)

        assert observable.to_dict()[Observable.VALUES_KEY] == before

    def test_charged_only_drops_neutral_constituents(self):
        observable = EEC(bin_edges=self.BIN_EDGES, pt_min=0.0, charged_only=True)

        class _Jet:
            def pt(self):
                return 100.0

            def constituents(self):
                return [
                    pseudo_jet(pt=50.0, eta=0.0, pythia_event_index=1, charged=True),
                    pseudo_jet(pt=50.0, eta=0.1, pythia_event_index=2, charged=False),
                ]

        observable.fill(jets=[_Jet()], cluster=None)

        # Only one charged constituent survives, so there is no pair to accumulate.
        assert observable.to_dict()[Observable.METADATA_KEY][EEC.W_PAIRS_KEY] == 0.0


class TestEECNormalizationConventions:

    def test_per_jet_conversion_uses_the_ordered_pair_factor(self):
        metadata = {EEC.W_PAIRS_KEY: 8.0, EEC.N_JETS_KEY: 4}
        shape = np.array([1.0, 2.0])

        converted = EEC.to_per_jet_normalized(shape, metadata)

        assert converted.tolist() == pytest.approx([4.0, 8.0])

    def test_normalization_denominator_switches_with_the_convention(self):
        metadata = {EEC.W_PAIRS_KEY: 8.0, EEC.N_JETS_KEY: 4}

        assert EEC.normalization_denominator(metadata, per_jet_normalized=False) == 8.0
        assert EEC.normalization_denominator(metadata, per_jet_normalized=True) == 4.0

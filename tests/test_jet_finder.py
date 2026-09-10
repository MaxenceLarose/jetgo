"""
    @file:              test_jet_finder.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       Tests for the jet finder, covering the clustering itself and the composition of the
                        kinematic selections a jet must pass.
"""

import math

import pytest

from jetgo.simulation import JetFinder

from conftest import pseudo_jet


class TestClustering:

    def test_two_separated_prongs_give_two_jets(self, two_prong_event):
        finder = JetFinder(radius=0.4)

        jets = finder.find_jets(finder.get_cluster(two_prong_event))

        assert len(jets) == 2

    def test_jets_come_back_sorted_by_decreasing_pt(self, two_prong_event):
        finder = JetFinder(radius=0.4)

        jets = finder.find_jets(finder.get_cluster(two_prong_event))

        assert [jet.pt() for jet in jets] == sorted((jet.pt() for jet in jets), reverse=True)

    def test_a_wide_radius_merges_back_to_back_prongs_into_one_jet(self, two_prong_event):
        finder = JetFinder(radius=4.0)

        jets = finder.find_jets(finder.get_cluster(two_prong_event))

        assert len(jets) == 1


class TestSelectors:

    def _cluster_and_select(self, finder, particles):
        return finder.find_jets(finder.get_cluster(particles))

    def test_no_cuts_keeps_everything(self, two_prong_event):
        assert len(self._cluster_and_select(JetFinder(radius=0.4), two_prong_event)) == 2

    def test_pt_min_removes_the_softer_jet(self, two_prong_event):
        jets = self._cluster_and_select(JetFinder(radius=0.4, pt_min=60.0), two_prong_event)

        assert len(jets) == 1
        assert jets[0].pt() > 60.0

    def test_pt_max_removes_the_harder_jet(self, two_prong_event):
        jets = self._cluster_and_select(JetFinder(radius=0.4, pt_max=60.0), two_prong_event)

        assert len(jets) == 1
        assert jets[0].pt() < 60.0

    def test_pt_min_and_pt_max_compose_into_a_window(self, two_prong_event):
        # The stand-in event clusters into jets of 80 and 50 GeV; only the harder one is in the window.
        inside = self._cluster_and_select(JetFinder(radius=0.4, pt_min=70.0, pt_max=90.0), two_prong_event)
        outside = self._cluster_and_select(JetFinder(radius=0.4, pt_min=200.0), two_prong_event)

        assert len(inside) == 1
        assert len(outside) == 0

    def test_pseudorapidity_cut_bites(self):
        particles = [
            pseudo_jet(pt=50.0, eta=0.0, phi=0.0, pythia_event_index=1),
            pseudo_jet(pt=50.0, eta=3.0, phi=math.pi, pythia_event_index=2),
        ]

        assert len(self._cluster_and_select(JetFinder(radius=0.4), particles)) == 2
        assert len(self._cluster_and_select(JetFinder(radius=0.4, max_abs_pseudorapidity=1.0), particles)) == 1

    def test_rapidity_cut_bites(self):
        particles = [
            pseudo_jet(pt=50.0, eta=0.0, phi=0.0, pythia_event_index=1),
            pseudo_jet(pt=50.0, eta=3.0, phi=math.pi, pythia_event_index=2),
        ]

        assert len(self._cluster_and_select(JetFinder(radius=0.4, max_abs_rapidity=1.0), particles)) == 1

    def test_an_empty_event_yields_no_jets(self):
        finder = JetFinder(radius=0.4, pt_min=10.0)

        assert len(finder.find_jets(finder.get_cluster([]))) == 0

"""
    @file:              test_kinematics.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       Tests for the shared kinematic helpers: the ΔR definition every observable and tagger
                        agrees on, and the user-index encoding that carries a particle's Pythia8 index and
                        charge through FastJet.
"""

import math

import pytest

from jetgo.kinematics import delta_r, encode_user_index, is_charged, pythia_index

from conftest import massive_pseudo_jet, pseudo_jet


class TestDeltaR:

    def test_pure_azimuthal_separation(self):
        a = pseudo_jet(pt=10.0, eta=0.0, phi=0.0)
        b = pseudo_jet(pt=10.0, eta=0.0, phi=0.3)

        assert delta_r(a, b) == pytest.approx(0.3)

    def test_pure_longitudinal_separation(self):
        a = pseudo_jet(pt=10.0, eta=0.0, phi=0.0)
        b = pseudo_jet(pt=10.0, eta=0.7, phi=0.0)

        assert delta_r(a, b) == pytest.approx(0.7)

    def test_combines_both_in_quadrature(self):
        a = pseudo_jet(pt=10.0, eta=0.0, phi=0.0)
        b = pseudo_jet(pt=10.0, eta=0.3, phi=0.4)

        assert delta_r(a, b) == pytest.approx(0.5)

    def test_is_symmetric(self):
        a = pseudo_jet(pt=10.0, eta=-0.2, phi=1.1)
        b = pseudo_jet(pt=30.0, eta=0.4, phi=2.3)

        assert delta_r(a, b) == pytest.approx(delta_r(b, a))

    def test_azimuthal_difference_wraps_across_pi(self):
        """Two particles straddling φ = 0 are 0.2 apart, not 2π - 0.2."""
        a = pseudo_jet(pt=10.0, eta=0.0, phi=0.1)
        b = pseudo_jet(pt=10.0, eta=0.0, phi=2 * math.pi - 0.1)

        assert delta_r(a, b) == pytest.approx(0.2)

    def test_azimuthal_difference_never_exceeds_pi(self):
        a = pseudo_jet(pt=10.0, eta=0.0, phi=0.0)
        b = pseudo_jet(pt=10.0, eta=0.0, phi=math.pi + 0.5)

        assert delta_r(a, b) == pytest.approx(math.pi - 0.5)

    def test_rapidity_and_pseudorapidity_agree_for_massless_particles(self):
        a = pseudo_jet(pt=10.0, eta=0.0, phi=0.0)
        b = pseudo_jet(pt=10.0, eta=0.6, phi=0.2)

        assert delta_r(a, b, use_pseudorapidity=False) == pytest.approx(
            delta_r(a, b, use_pseudorapidity=True)
        )

    def test_rapidity_and_pseudorapidity_differ_for_massive_particles(self):
        """The whole reason the flag exists: with mass, y and η are different quantities."""
        a = massive_pseudo_jet(pt=10.0, eta=0.0, phi=0.0, mass=5.0)
        b = massive_pseudo_jet(pt=10.0, eta=1.0, phi=0.0, mass=5.0)

        by_rapidity = delta_r(a, b, use_pseudorapidity=False)
        by_pseudorapidity = delta_r(a, b, use_pseudorapidity=True)

        assert by_pseudorapidity == pytest.approx(1.0)
        assert by_rapidity < by_pseudorapidity


class TestUserIndexEncoding:

    def test_charged_particle_keeps_a_positive_index(self):
        assert encode_user_index(7, charged=True) == 7

    def test_neutral_particle_gets_a_negative_index(self):
        assert encode_user_index(7, charged=False) == -7

    @pytest.mark.parametrize("charged", [True, False])
    def test_round_trips_through_the_readers(self, charged):
        particle = pseudo_jet(pt=10.0, pythia_event_index=42, charged=charged)

        assert pythia_index(particle) == 42
        assert is_charged(particle) is charged

    @pytest.mark.parametrize("bad_index", [0, -1])
    def test_rejects_indices_whose_sign_would_be_meaningless(self, bad_index):
        """Index 0 is Pythia8's placeholder; its sign cannot encode a charge."""
        with pytest.raises(ValueError, match="strictly positive"):
            encode_user_index(bad_index, charged=True)

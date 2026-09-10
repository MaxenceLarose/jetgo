"""
    @file:              particle_selector.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 07/2026

    @Description:       This file defines the ParticleSelector class, which selects particles from Pythia8 events for
                        jet reconstruction. It keeps final-state, visible particles, applies a rapidity acceptance
                        cut, and converts the survivors into FastJet PseudoJet objects.
"""

from __future__ import annotations

from typing import List, Optional

import fastjet as fj

from .._pythia import pythia8
from ..kinematics import encode_user_index


class ParticleSelector:
    """
    An object that is used to apply particle-level selections to a Pythia event and converts the accepted particles into
    FastJet PseudoJet objects.

    The selector retains only final-state, visible particles within a specified rapidity acceptance.
    """

    def __init__(
            self,
            max_abs_rapidity: Optional[float | int] = None,
            max_abs_pseudorapidity: Optional[float | int] = None,
            charged_only: bool = False
    ) -> None:
        """
        Constructor for ParticleSelector.

        Parameters
        ----------
        max_abs_rapidity : Optional[float | int]
            Maximum absolute rapidity ``|y|`` for accepted particles.
        max_abs_pseudorapidity: Optional[float | int]
            Maximum absolute pseudorapidity |𝜂| for accepted particles.
        charged_only : bool, default=False
            If True, drop neutral particles before jet clustering, so downstream jets are "charged-particle
            jets" (built entirely from charged tracks, e.g. as in ALICE analyses) rather than full-particle jets.
            This is a different knob from an observable's own ``charged_only`` flag (e.g. ``EEC``), which only
            filters constituent pairs *after* the jet -- including its axis and pT -- has already been built
            from whatever particles were selected here.
        """
        self._max_abs_rapidity = max_abs_rapidity
        self._max_abs_pseudorapidity = max_abs_pseudorapidity
        self._charged_only = charged_only

    def select(
            self,
            event: pythia8.Event
    ) -> List[fj.PseudoJet]:
        """
        Select particles from a Pythia event and convert them to a list of FastJet PseudoJets.

        The following selections are applied:
          - Final-state particles only;
          - Visible particles only (interacting via electromagnetic or strong force);
          - Charged particles only, if ``charged_only`` was set;
          - ``|y|`` < max_abs_rapidity.

        The Pythia8 index of each selected particle is encoded into the PseudoJet's ``user_index()`` by
        ``jetgo.kinematics.encode_user_index``, which defines the convention: its magnitude is
        the particle's index in ``event``, and its sign indicates whether the particle is charged (positive) or
        neutral (negative). Real final-state particles always have a positive index (index 0 in a Pythia8 event is a
        reserved placeholder, never a real particle), so this encoding is unambiguous and reversible.

        Parameters
        ----------
        event : pythia8.Event
            A Pythia-generated event.

        Returns
        -------
        particles : List[fj.PseudoJet]
            Selected particles as FastJet PseudoJets, ready for jet clustering.
        """
        particles: list[fj.PseudoJet] = []
        for index, particle in enumerate(event):
            if not particle.isFinal():
                continue

            if index == 0:
                raise RuntimeError(
                    "A final-state particle was found at Pythia8 event index 0. Index 0 is reserved for Pythia8's "
                    "internal event-record placeholder and should never be a real, final-state particle -- this "
                    "should not happen. Please investigate the event record before proceeding, since downstream "
                    "code relies on index 0 being an impossible, unambiguous sentinel for 'no particle'."
                )

            if not particle.isVisible():
                continue

            if self._charged_only and not particle.isCharged():
                continue

            if self._max_abs_pseudorapidity and abs(particle.eta()) > self._max_abs_pseudorapidity:
                    continue

            if self._max_abs_rapidity and abs(particle.y()) > self._max_abs_rapidity:
                    continue

            pseudo_jet = fj.PseudoJet(particle.px(), particle.py(), particle.pz(), particle.e())
            pseudo_jet.set_user_index(encode_user_index(index, particle.isCharged()))
            particles.append(pseudo_jet)

        return particles

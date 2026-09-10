"""
    @file:              event_generator.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 06/2026

    @Description:       This file defines the EventGenerator class, a wrapper around Pythia8 for generating hard QCD
                        collision events with configurable beams, center-of-mass energy, and partonic phase space cuts.
"""

from __future__ import annotations

from .._pythia import pythia8


class EventGenerator:
    """
    Pythia8 event generator for hard-QCD hadronic collision events.

    This class configures Pythia8 for generic beam–beam collisions, restricts the hard-scattering phase space in pT-hat,
    and provides utilities for event generation and cross-section normalization.
    """

    def __init__(
        self,
        beam_1: int,
        beam_2: int,
        sqrt_s: float | int,
        pt_min: float | int,
        pt_max: float | int,
        random_seed: int = 0,
    ) -> None:
        """
        Initialize and configure the Pythia8 event generator.

        Parameters
        ----------
        beam_1 : int
            PDG ID of beam A (e.g. 2212 = proton).
        beam_2 : int
            PDG ID of beam B (e.g. 2212 = proton).
        sqrt_s : float | int
            Center-of-mass energy in GeV.
        pt_min : float | int
            Lower bound of the partonic transverse momentum (p̂T) in GeV.
        pt_max : float | int
            Upper bound of the partonic transverse momentum (p̂T) in GeV.
        random_seed : int
            Seed for Pythia's RNG (0 lets Pythia choose a seed automatically).
        """
        pythia = pythia8.Pythia()

        pythia.readString(f"Beams:idA = {beam_1}")
        pythia.readString(f"Beams:idB = {beam_2}")
        pythia.readString(f"Beams:eCM = {sqrt_s:.1f}")

        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {random_seed}")

        pythia.readString("HardQCD:all = on")

        pythia.readString(f"PhaseSpace:pTHatMin = {pt_min:.1f}")
        pythia.readString(f"PhaseSpace:pTHatMax = {pt_max:.1f}")

        pythia.init()

        self._pythia = pythia

    def __call__(
            self,
            n_events: int
    ) -> pythia8.Event:
        """
        Generate events from the configured Pythia8 instance.

        Parameters
        ----------
        n_events : int
            Number of events that needs to be generated.

        Yields
        ------
        pythia8.Event
            Pythia8 event.
        """
        generated = 0

        while generated < n_events:
            self._pythia.next()

            generated += 1
            yield self._pythia.event

    @property
    def scale_factor(self) -> float:
        """
        Compute the event normalization factor for cross-section scaling.

        In Pythia, events generated with restricted pT-hat ranges represent a biased Monte Carlo sample. Each event must
        therefore be reweighted to recover the physical cross-section.

        The normalization factor is defined as:

            scale_factor = σ_gen / Σw

        where:
            - σ_gen is the Pythia estimate of the process cross-section;
            - Σw is the sum of event weights in the generated sample.

        Returns
        -------
        scale_factor: float
            Normalization factor in nanobarns (nb).
        """
        info = self._pythia.infoPython()
        return info.sigmaGen() / info.weightSum() * 1e6

    def print_statistics(self) -> None:
        """
        Print Pythia run statistics, including cross-section estimates, event counts, and generation efficiency.
        """
        self._pythia.stat()

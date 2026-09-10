"""
    @file:              conftest.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       Shared fixtures and stand-ins for the test suite.

                        Most of jetgo can be exercised without Pythia8: the simulation loop only ever asks its
                        event generator for events, its particle selector for PseudoJets, and its taggers for a
                        flavor. Substituting all three lets the whole pipeline, including the output file, be
                        tested on a machine with no generator installed. The tests that genuinely need Pythia8
                        are marked ``pythia`` and skipped when it is absent.
"""

import math
from typing import List, Optional

import fastjet as fj
import pytest

from jetgo.kinematics import encode_user_index
from jetgo.taggers.base import JetFlavorTagger
from jetgo.taggers.flavor import JetFlavor


def pseudo_jet(
    pt: float,
    eta: float = 0.0,
    phi: float = 0.0,
    pythia_event_index: int = 1,
    charged: bool = True,
) -> fj.PseudoJet:
    """
    Build a massless PseudoJet from cylindrical coordinates, with the user index this package expects.

    Being massless, its rapidity and its pseudorapidity coincide, so a test that wants the two to differ must
    build a massive particle instead (see ``massive_pseudo_jet``).
    """
    particle = fj.PseudoJet(
        pt * math.cos(phi),
        pt * math.sin(phi),
        pt * math.sinh(eta),
        pt * math.cosh(eta),
    )
    particle.set_user_index(encode_user_index(pythia_event_index, charged))
    return particle


def massive_pseudo_jet(pt: float, eta: float, phi: float, mass: float) -> fj.PseudoJet:
    """
    Build a PseudoJet with non-zero mass, whose rapidity therefore differs from its pseudorapidity.
    """
    pz = pt * math.sinh(eta)
    energy = math.sqrt(pt ** 2 + pz ** 2 + mass ** 2)
    return fj.PseudoJet(pt * math.cos(phi), pt * math.sin(phi), pz, energy)


class StubEventGenerator:
    """
    Stands in for ``EventGenerator``: yields opaque event objects and reports a fixed scale factor.

    The events are never inspected, because the stub taggers below decide a flavor without looking at them.
    """

    def __init__(self, n_events: int, scale_factor: float = 2.0) -> None:
        self._n_events = n_events
        self._scale_factor = scale_factor
        self.printed_statistics = False

    def __call__(self, n_events: int):
        for index in range(min(n_events, self._n_events)):
            yield f"event-{index}"

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    def print_statistics(self) -> None:
        self.printed_statistics = True


class StubParticleSelector:
    """
    Stands in for ``ParticleSelector``: returns the same constituents for every event.
    """

    def __init__(self, particles: List[fj.PseudoJet]) -> None:
        self._particles = particles

    def select(self, event) -> List[fj.PseudoJet]:
        return list(self._particles)


class StubTagger(JetFlavorTagger):
    """
    Stands in for a real tagger: hands out a caller-supplied sequence of flavors, cycling through it.

    ``identifier`` is a plain string rather than a ``JetFlavorTaggerIdentifier`` member, since the enum names
    only the strategies the package actually ships and the simulator only ever uses the identifier as a key.
    """

    def __init__(self, name: str, flavors: List[JetFlavor], diagnostics: Optional[dict] = None) -> None:
        self._name = name
        self._flavors = flavors
        self._diagnostics = diagnostics or {}
        self._calls = 0

    @property
    def identifier(self) -> str:
        return self._name

    def get_diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def tag(self, jet: fj.PseudoJet, event) -> JetFlavor:
        flavor = self._flavors[self._calls % len(self._flavors)]
        self._calls += 1
        return flavor


@pytest.fixture
def two_prong_event() -> List[fj.PseudoJet]:
    """
    Two well-separated clusters of particles, which anti-kT with R = 0.4 resolves into exactly two jets.
    """
    particles = []
    index = 1
    for eta_center, phi_center, pts in ((0.0, 0.0, (60.0, 20.0)), (0.0, math.pi, (40.0, 10.0))):
        for offset, pt in enumerate(pts):
            particles.append(
                pseudo_jet(pt, eta=eta_center + 0.05 * offset, phi=phi_center + 0.05 * offset,
                           pythia_event_index=index, charged=(index % 2 == 1))
            )
            index += 1
    return particles

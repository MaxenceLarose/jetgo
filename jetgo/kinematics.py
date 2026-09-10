"""
    @file:              kinematics.py
    @Author:            Maxence Larose

    @Creation Date:     07/2026
    @Last modification: 09/2026

    @Description:       This file defines the shared kinematic helpers used across the package: the ``delta_r``
                        function, so every part of the codebase that needs a ΔR (the EEC observable, a tagger
                        matching a jet to a parton, ...) agrees on the same definition, and the encoding of a
                        particle's Pythia8 index and charge into a ``PseudoJet``'s user index.

                        FastJet's ``PseudoJet`` carries no particle identity of its own, only a four-momentum and
                        a single integer slot, ``user_index()``. This package uses that slot to carry two facts
                        about the particle a ``PseudoJet`` was built from: its position in the Pythia8 event
                        record, and whether it is electrically charged. The magnitude is the Pythia8 index and
                        the sign is the charge, positive for charged and negative for neutral.

                        That encoding works because index 0 in a Pythia8 event record is a reserved placeholder
                        and never a real particle, so no real particle has an index whose sign is meaningless.
                        ``ParticleSelector`` writes the encoding and observables read it back. Anyone writing
                        their own ``Observable`` subclass should read it through the functions below rather than
                        touching ``user_index()`` directly, since the encoding is an implementation detail that
                        these three functions define.
"""

import math

import fastjet as fj


def delta_r(
    a: fj.PseudoJet,
    b: fj.PseudoJet,
    use_pseudorapidity: bool = False
) -> float:
    """
    Angular separation ΔR between two four-vectors.

    Parameters
    ----------
    a : fj.PseudoJet
        First four-vector.
    b : fj.PseudoJet
        Second four-vector.
    use_pseudorapidity : bool, default=False
        If True, ΔR is computed using the pseudorapidity η instead of the (true, mass-dependent) rapidity y, i.e.
        ΔR = sqrt((Δη)² + (Δφ)²) instead of ΔR = sqrt((Δy)² + (Δφ)²).

    Returns
    -------
    delta_r : float
        ΔR = sqrt((Δy or Δη)² + (Δφ)²), with Δφ correctly wrapped to [0, π].
    """
    if use_pseudorapidity:
        d_y = a.eta() - b.eta()
    else:
        d_y = a.rap() - b.rap()

    d_phi = a.delta_phi_to(b)

    return math.sqrt(d_y ** 2 + d_phi ** 2)


def encode_user_index(
    pythia_event_index: int,
    charged: bool
) -> int:
    """
    Encode a particle's Pythia8 event-record index and charge into a single ``PseudoJet`` user index.

    Parameters
    ----------
    pythia_event_index : int
        Position of the particle in the Pythia8 event record. Must be strictly positive: index 0 is Pythia8's
        reserved placeholder entry and never a real particle.
    charged : bool
        Whether the particle is electrically charged.

    Returns
    -------
    user_index : int
        The index itself if the particle is charged, its negation if it is neutral.

    Raises
    ------
    ValueError
        If ``pythia_event_index`` is not strictly positive, which would make the sign meaningless.
    """
    if pythia_event_index <= 0:
        raise ValueError(
            f"Expected a strictly positive Pythia8 event-record index, got {pythia_event_index}. Index 0 is "
            f"Pythia8's reserved placeholder and should never belong to a real particle, and the sign of the "
            f"user index is reserved for the particle's charge."
        )

    return pythia_event_index if charged else -pythia_event_index


def is_charged(particle: fj.PseudoJet) -> bool:
    """
    Whether a particle is electrically charged, read back from its encoded user index.

    Parameters
    ----------
    particle : fj.PseudoJet
        A particle whose user index was written by ``encode_user_index``, i.e. one produced by
        ``ParticleSelector``.

    Returns
    -------
    charged : bool
        True if the particle is charged.
    """
    return particle.user_index() > 0


def pythia_index(particle: fj.PseudoJet) -> int:
    """
    The particle's position in the Pythia8 event record, read back from its encoded user index.

    Parameters
    ----------
    particle : fj.PseudoJet
        A particle whose user index was written by ``encode_user_index``, i.e. one produced by
        ``ParticleSelector``.

    Returns
    -------
    index : int
        Position of the particle in the Pythia8 event record it was selected from.
    """
    return abs(particle.user_index())

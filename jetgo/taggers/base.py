"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 07/2026

    @Description:       This file defines the JetFlavorTagger abstract base class. It provides the shared interface
                        and common PDG-id-to-flavor mapping used by all concrete jet flavor tagging strategies.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import fastjet as fj

from .flavor import JetFlavor
from .identifier import JetFlavorTaggerIdentifier

if TYPE_CHECKING:
    from .._pythia import pythia8


class JetFlavorTagger(ABC):
    """
    Abstract interface for jet quark/gluon tagging strategies.

    Implementations label a reconstructed jet as originating from a quark or a gluon (or leave it untagged) using
    information from the full Pythia8 event record for the event the jet was reconstructed from.
    """

    HARD_PROCESS_STATUS = 23

    QUARK_PDG_IDS = frozenset(range(1, 7))
    GLUON_PDG_ID = 21

    @property
    @abstractmethod
    def identifier(self) -> JetFlavorTaggerIdentifier:
        """
        Return the identifier of this tagging strategy. Used as a key to distinguish this tagger's results when
        several taggers are saved to the same output file.

        Returns
        -------
        identifier : JetFlavorTaggerIdentifier
            Identifier of the tagging strategy.
        """

    def _get_flavor_from_pdg_id(self, pdg_id: int) -> JetFlavor:
        """
        Map a PDG particle id to a flavor (quark or gluon).

        Parameters
        ----------
        pdg_id : int
            PDG Monte Carlo particle id, as returned by ``Particle.id()``.

        Returns
        -------
        flavor : JetFlavor
            - ``JetFlavor.QUARK`` for a quark or antiquark (any of the 6 flavors, either sign of PDG id);
            - ``JetFlavor.GLUON`` for a gluon;
            - ``JetFlavor.UNTAGGED`` for anything else (i.e. not a hard-scattering parton).
        """
        abs_id = abs(pdg_id)

        if abs_id in self.QUARK_PDG_IDS:
            return JetFlavor.QUARK

        if abs_id == self.GLUON_PDG_ID:
            return JetFlavor.GLUON

        return JetFlavor.UNTAGGED

    def get_diagnostics(self) -> dict:
        """
        Return tagger-specific diagnostic counters accumulated since construction, to be included in the run's
        saved metadata and printed summary alongside the quark/gluon/untagged counts. Empty by default; a
        subclass that tracks something extra (e.g. how often it fell back to a secondary strategy) should
        override this.

        Returns
        -------
        dict
            Extra diagnostic counters, or an empty dict if this tagger doesn't track any.
        """
        return {}

    @abstractmethod
    def tag(
        self,
        jet: fj.PseudoJet,
        event: "pythia8.Event"
    ) -> JetFlavor:
        """
        Label a jet as originating from a quark or a gluon.

        Parameters
        ----------
        jet : fj.PseudoJet
            Reconstructed jet to tag.
        event : pythia8.Event
            Full Pythia8 event record for the same event the jet was reconstructed from. Must be the untouched event
            yielded by the generator, since it needs to still contain the original hard-scattering partons and, for
            ancestry-based strategies, the full particle history.

        Returns
        -------
        flavor : JetFlavor
            ``JetFlavor.QUARK`` or ``JetFlavor.GLUON`` if the jet was successfully tagged, ``JetFlavor.UNTAGGED``
            otherwise.
        """
        raise NotImplementedError

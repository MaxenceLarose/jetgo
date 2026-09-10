"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 07/2026

    @Description:       This file defines the Observable abstract base class. It provides the shared interface and
                        common behavior that all physics observables inherit.

                        Observables accumulate a histogram during the simulation, on bin edges fixed when the
                        observable is constructed. Those edges are not a property of the physics being
                        simulated -- they come from outside it, from whichever measurement or analysis the run
                        will be read against -- so every observable takes them as a required argument rather
                        than choosing any itself.

                        Saving each contribution unbinned instead would leave that choice open until analysis
                        time, which is the more flexible design and was the original one. It does not survive
                        contact with the run sizes this project needs: an EEC stores one entry per constituent
                        *pair*, which grows quadratically with jet multiplicity, and a million-event window
                        costs gigabytes of RAM and JSON to keep values that are only ever histogrammed onto one
                        known binning anyway. Binning at fill time makes the cost independent of the run
                        length.

                        What remains downstream (see ``finalize``) is the normalization, plus whatever
                        re-binning each observable implements on top of the saved edges. Only the
                        cross-section does: it sums whole bins onto coarser ones, and its edges are
                        deliberately filled fine enough to be a refinement of every binning the analysis asks
                        for. The EEC serves only the edges it was filled on.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import fastjet as fj
import numpy as np

from .identifier import ObservableIdentifier


class Observable(ABC):
    """
    Abstract base class for physics observables.

    Defines the shared interface and delegates observable-specific logic to abstract methods implemented by each
    concrete subclass.
    """

    METADATA_KEY = "metadata"
    VALUES_KEY = "values"

    @property
    @abstractmethod
    def identifier(self) -> ObservableIdentifier:
        """
        Return the identifier of the observable. The identifier should match the observable identifier defined on the
        HEP data website.

        Returns
        -------
        identifier : ObservableIdentifier
            Identifier of the observable.
        """

    @abstractmethod
    def fill(
            self,
            jets: List[fj.PseudoJet],
            cluster: fj.ClusterSequence
    ) -> None:
        """
        Extract this observable's contributions from one event and accumulate them into its histogram. Called
        once per generated event inside the simulation loop. Subclasses decide which quantities to extract and
        which bin each one lands in.

        Parameters
        ----------
        jets : List[fj.PseudoJet]
            Reconstructed jets for this event, as returned by the jet finder.
        cluster : fj.ClusterSequence
            Cluster sequence for this event, as returned by the cluster finder.
        """

    @abstractmethod
    def scale(
            self,
            scale_factor: float | int
    ) -> None:
        """
        Record the event-normalization scale factor, applied later, at ``finalize`` time.

        Parameters
        ----------
        scale_factor : float
            Factor by which every bin will eventually be multiplied (``sigmaGen / weightSum`` from the generator).
        """

    @classmethod
    @abstractmethod
    def finalize(cls, values, bin_edges: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Normalize the histogram saved by ``to_dict`` onto ``bin_edges``.

        Called downstream (by ``ObservableVisualizer`` or analysis code) -- never during the simulation itself.
        ``bin_edges`` cannot reopen the binning decision, which was made when the observable was constructed:
        it is the caller stating which edges it believes it is reading, and each subclass either serves it from
        the edges it was filled on or raises. Implementations must never silently return values on edges other
        than the ones asked for.

        Parameters
        ----------
        values
            The saved histogram, in whatever shape this observable writes it (see each subclass's ``to_dict``).
        bin_edges : np.ndarray
            The bin edges the result is wanted on.
        metadata : Dict
            This observable's saved metadata (see ``_get_complementary_metadata``).

        Returns
        -------
        np.ndarray
            Normalized values, one per target bin.
        """

    @abstractmethod
    def _get_complementary_metadata(self) -> Dict:
        """
        Return complementary metadata for this observable.

        Returns
        -------
        complementary_metadata : Dict
            Additional metadata needed for that specific observable.
        """

    @abstractmethod
    def _get_binned_values(self):
        """
        Return this observable's accumulated histogram (see ``fill``), one value per bin of the edges it was
        constructed with.
        """

    def to_dict(self) -> Dict:
        """
        Return this observable's accumulated histogram and metadata as a JSON-serializable dict.

        Returns
        -------
        Dict
            Dictionary with keys ``VALUES_KEY`` (one value per bin) and ``METADATA_KEY`` (observable-specific
            metadata, which always includes the bin edges the values sit on).
        """
        return {
            self.VALUES_KEY: self._get_binned_values(),
            self.METADATA_KEY: self._get_complementary_metadata(),
        }

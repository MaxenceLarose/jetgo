"""
    @file:              double_differential_jet_cross_section.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 07/2026

    @Description:       This file defines the DoubleDifferentialJetCrossSection observable, the double
                        differential inclusive jet cross-section d²σ/dpT dy. Jets are accumulated into a pT
                        histogram on the bin edges the observable is constructed with (see ``Observable``);
                        unlike the EEC, that grid can still be coarsened downstream, by summing whole bins
                        (see ``finalize``).
"""

from typing import Dict, List, Optional

import fastjet as fj
import numpy as np

from .base import Observable
from .identifier import ObservableIdentifier


class DoubleDifferentialJetCrossSection(Observable):
    """
    Double-differential inclusive jet cross-section d²σ/dpT dy (or d²σ/dpT dη, if jets were selected in
    pseudorapidity instead -- see ``__init__``).
    """

    RAPIDITY_INTERVAL_KEY = "rapidity_interval"
    SCALE_FACTOR_KEY = "scale_factor"
    # The pT grid the values were accumulated on. Always written, and what ``finalize`` re-bins from -- a file
    # without it predates forced binning and holds individual jet pT values instead.
    BIN_EDGES_KEY = "bin_edges"

    def __init__(
        self,
        bin_edges: np.ndarray,
        max_abs_rapidity: Optional[float | int] = None,
        max_abs_pseudorapidity: Optional[float | int] = None,
    ) -> None:
        """
        Constructor for the JetCrossSection class.

        Parameters
        ----------
        bin_edges : np.ndarray
            The jet pT grid to accumulate the cross-section on. Required, so that the file size stays
            independent of the run length and consistent with the EEC, which has no choice in the matter (see
            ``Observable``).

            The edges must be a *refinement* of every binning you will later ask for. One analysis may
            finalize over a single bin ``[pt_min, pt_max]`` while another wants five equal sub-bins, so a
            grid dividing both reproduces them exactly. ``finalize`` re-bins by summing whole bins and
            raises if the requested edges do not fall on this grid.
        max_abs_rapidity : Optional[float | int]
            Maximum absolute rapidity |y| used in the jet selection. Exactly one of ``max_abs_rapidity``/
            ``max_abs_pseudorapidity`` must be given -- whichever cut ``JetFinder`` was actually configured
            with -- since the acceptance interval width (Δy or Δη = 2 * the given value, assuming a symmetric
            acceptance around mid-(pseudo)rapidity) is what the cross-section is normalized by.
        max_abs_pseudorapidity : Optional[float | int]
            Maximum absolute pseudorapidity |η| used in the jet selection, if that was the cut applied instead
            of rapidity (e.g. matching a charged-particle-jet measurement).
        """
        if (max_abs_rapidity is None) == (max_abs_pseudorapidity is None):
            raise ValueError(
                "Exactly one of max_abs_rapidity or max_abs_pseudorapidity must be given, matching whichever "
                "cut JetFinder was actually configured with."
            )

        acceptance = max_abs_rapidity if max_abs_rapidity is not None else max_abs_pseudorapidity
        self._rapidity_interval = 2 * acceptance
        self._scale_factor = 1.0
        self._bin_edges = np.asarray(bin_edges, dtype=float)
        self._counts = np.zeros(len(self._bin_edges) - 1, dtype=float)

    @property
    def identifier(self) -> ObservableIdentifier:
        """
        Identifier of the observable.

        Returns
        -------
        identifier : ObservableIdentifier
            Identifier of the observable.
        """
        return ObservableIdentifier.D2SIG_DPT_DYRAP

    def scale(
            self,
            scale_factor: float | int
    ) -> None:
        """
        Record the scale factor converting raw jet counts into an absolute cross-section in nb, applied later at
        ``finalize`` time.

        Parameters
        ----------
        scale_factor : float
            Ratio sigmaGen / weightSum returned by the generator.
        """
        self._scale_factor = scale_factor

    def fill(
            self,
            jets: List[fj.PseudoJet],
            cluster: fj.ClusterSequence
    ) -> None:
        """
        Accumulate every jet reconstructed in one event into its transverse momentum bin. Jets falling outside
        the grid are dropped, exactly as they would be when histogramming stored pT values after the fact.

        Parameters
        ----------
        jets : List[fj.PseudoJet]
            Reconstructed jets for this event, as returned by the jet finder after kinematic selections.
        cluster : fj.ClusterSequence
            Cluster sequence for this event. Unused by this observable.
        """
        for jet in jets:
            index = int(np.searchsorted(self._bin_edges, jet.pt(), side="right")) - 1
            if 0 <= index < len(self._counts):
                self._counts[index] += 1.0

    @classmethod
    def finalize(cls, values: List[float | int], bin_edges: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Sum the accumulated jet counts onto ``bin_edges`` and normalize into d²σ/dpT dy, by dividing by each
        bin's width in pT and by the total rapidity interval Δy.

        Parameters
        ----------
        values : List[float]
            The jet count per bin of the grid the observable was filled on.
        bin_edges : np.ndarray
            The bin edges the result is wanted on. They must fall on that grid, so that each requested bin is a
            whole number of grid bins.
        metadata : Dict
            This observable's saved metadata, must contain ``RAPIDITY_INTERVAL_KEY``, ``SCALE_FACTOR_KEY`` and
            ``BIN_EDGES_KEY``.

        Returns
        -------
        np.ndarray
            d²σ/dpT dy in nb/GeV, one value per target bin.

        Raises
        ------
        ValueError
            If the requested edges do not lie on the filled grid, so the requested bins cannot be formed by
            summing whole grid bins.
        """
        bin_edges = np.asarray(bin_edges, dtype=float)

        if cls.BIN_EDGES_KEY not in metadata:
            raise ValueError(
                "This cross section holds individual jet pT values rather than a pT histogram, which no "
                "version of this observable still produces. Regenerate the simulation, which will accumulate "
                "it directly on a grid the requested edges can be summed from."
            )

        counts = cls._rebin(np.asarray(values, dtype=float),
                            np.asarray(metadata[cls.BIN_EDGES_KEY], dtype=float), bin_edges)

        bin_widths = np.diff(bin_edges)
        rapidity_interval = metadata[cls.RAPIDITY_INTERVAL_KEY]
        scale_factor = metadata[cls.SCALE_FACTOR_KEY]

        return counts * scale_factor / (bin_widths * rapidity_interval)

    @staticmethod
    def _rebin(counts: np.ndarray, filled_edges: np.ndarray, requested_edges: np.ndarray) -> np.ndarray:
        """
        Sum a histogram's bins onto coarser edges that lie on its own grid.

        Parameters
        ----------
        counts : np.ndarray
            Counts on ``filled_edges``.
        filled_edges : np.ndarray
            The grid the observable was filled on.
        requested_edges : np.ndarray
            The coarser edges wanted, each of which must coincide with one of ``filled_edges``.

        Returns
        -------
        np.ndarray
            Counts on ``requested_edges``.

        Raises
        ------
        ValueError
            If any requested edge does not lie on the filled grid.
        """
        indices = []
        for edge in requested_edges:
            match = np.flatnonzero(np.isclose(filled_edges, edge, rtol=0.0, atol=1e-9))
            if match.size == 0:
                raise ValueError(
                    f"Requested bin edge {edge:.6g} does not lie on the grid this cross section was filled on "
                    f"({len(filled_edges) - 1} bins spanning [{filled_edges[0]:.6g}, {filled_edges[-1]:.6g}]). "
                    f"The individual jet pT values are no longer there, so this bin cannot be formed."
                )
            indices.append(int(match[0]))

        return np.array([counts[lo:hi].sum() for lo, hi in zip(indices[:-1], indices[1:])], dtype=float)

    def _get_complementary_metadata(self) -> dict:
        """
        Return complementary metadata.

        Returns
        -------
        complementary_metadata : dict
            Additional metadata.
        """
        return {
            self.RAPIDITY_INTERVAL_KEY: self._rapidity_interval,
            self.SCALE_FACTOR_KEY: self._scale_factor,
            self.BIN_EDGES_KEY: self._bin_edges.tolist(),
        }

    def _get_binned_values(self) -> List[float]:
        """
        Return the jet count accumulated in each pT bin so far.

        Returns
        -------
        List[float]
            The jet count per pT bin.
        """
        return self._counts.tolist()

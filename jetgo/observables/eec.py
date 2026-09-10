"""
    @file:              eec.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 07/2026

    @Description:       This file defines the EEC observable, the Energy-Energy Correlator as a function of the
                        angular separation ΔR between pairs of jet constituents. Every constituent pair is
                        accumulated into a ΔR histogram on the bin edges the observable is constructed with --
                        necessarily so, since one entry per *pair* is quadratic in jet multiplicity and storing
                        them individually is what makes a large run unaffordable (see ``Observable``).

                        Each pair is weighted by (z_i * z_j)^n, where z_i = pT_i / pT_jet is the constituent's
                        momentum fraction of its own jet -- not by the raw (pT_i * pT_j)^n, so that a pair's
                        weight means the same thing regardless of the absolute pT of the jet it came from.
                        That is what lets a per-jet-normalized EEC remain well defined across jets of
                        different pT within one window. ΔR is computed manually from the rapidity (or,
                        optionally, the pseudorapidity) and the azimuthal angle of each constituent.
"""

from typing import Dict, List

import fastjet as fj
import numpy as np

from .base import Observable
from .identifier import ObservableIdentifier
from ..kinematics import delta_r, is_charged


class EEC(Observable):
    """
    Energy-Energy Correlator (EEC) as a function of angular separation ΔR.

    Defined as (Eq. 1):

        EEC(ΔR) = (1 / W_pairs) * (1 / ΔR_bin) * Σ_{jets} Σ_{i<j ∈ jet} (z_i * z_j)^n

    where z_i = pT_i / pT_jet is constituent i's momentum fraction of its own jet, and W_pairs is the total sum
    of all pair weights,

        W_pairs = Σ_{jets} Σ_{i<j ∈ jet} (z_i * z_j)^n,

    such that the observable is normalized to unity after integration over ΔR. Optionally, only charged constituents
    and/or constituents above a minimum transverse momentum threshold can be included.
    """

    W_PAIRS_KEY = "w_pairs"
    N_JETS_KEY = "n_jets"
    # The ΔR edges the values were accumulated on. Always written, and what ``finalize`` checks the requested
    # edges against -- a file without it predates forced binning and holds raw pairs instead.
    BIN_EDGES_KEY = "bin_edges"

    # ``fill`` accumulates each unordered constituent pair {i, j} exactly once (i < j), which is all the
    # unity-normalized convention above needs -- any per-pair constant cancels between the numerator and
    # W_pairs. The published EEC convention instead sums over *ordered* pairs, so that summing the weights
    # over every pair of a jet gives (Σ_i z_i)^n = 1 rather than roughly half that. That factor does not
    # cancel once the observable is renormalized per jet instead of to unity, so it is applied there (see
    # ``to_per_jet_normalized``). Self-pairs (i = j), also part of that convention, sit entirely at ΔR = 0
    # and so never enter a ΔR bin above zero.
    ORDERED_PAIR_FACTOR = 2

    def __init__(
        self,
        bin_edges: np.ndarray,
        pt_min: float | int = 0,
        energy_weight: int = 1,
        charged_only: bool = False,
        use_pseudorapidity: bool = False
    ) -> None:
        """
        Constructor for the EEC observable.

        Parameters
        ----------
        bin_edges : np.ndarray
            The ΔR bin edges to accumulate the correlator on, ordinarily those of the measurement the run will
            be compared against. Required: the number of constituent pairs is quadratic in jet multiplicity, so
            keeping them individually in order to leave this choice open until analysis time costs gigabytes on
            a run of any size (see ``Observable``).

            Since the pairs are not kept, these must be the edges the analysis will ask ``finalize`` for; it
            raises rather than silently re-binning if they are not. Pairs falling outside them still count
            towards W_pairs, so the normalization does not depend on the range chosen.
        pt_min : float | int, default=0
            Minimum constituent transverse momentum pT [GeV] required for a
            particle to contribute to the correlator.
        energy_weight : int, default=1
            Exponent n used in the pair weight (pT_i · pT_j)^n.
        charged_only : bool, default=False
            If True, only charged jet constituents are included in the
            correlator.
        use_pseudorapidity : bool, default=False
            If True, ΔR is computed using the pseudorapidity η instead of the (true, mass-dependent) rapidity y, i.e.
            ΔR = sqrt((Δη)² + (Δφ)²) instead of ΔR = sqrt((Δy)² + (Δφ)²).
        """
        self._energy_weight = energy_weight
        self._w_pairs = 0
        self._n_jets = 0
        self._charged_only = charged_only
        self._pt_min = pt_min
        self._use_pseudorapidity = use_pseudorapidity
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
        return ObservableIdentifier.EEC

    def fill(
        self,
        jets: List[fj.PseudoJet],
        cluster: fj.ClusterSequence
    ) -> None:
        """
        Accumulate every constituent pair of every reconstructed jet from one event into the ΔR histogram.

        For each jet, constituents are first filtered according to the minimum constituent pT requirement and,
        optionally, the charged-particle selection. All unique constituent pairs (i < j) are then considered and
        added to their ΔR bin, along with the total pair weight W_pairs accumulated for later normalization.

        Parameters
        ----------
        jets : List[fj.PseudoJet]
            Reconstructed jets for this event.
        cluster : fj.ClusterSequence
            Cluster sequence for this event. Unused by this observable.
        """
        self._n_jets += len(jets)

        for jet in jets:

            constituents = []
            for c in jet.constituents():
                if c.pt() <= self._pt_min:
                    continue
                if self._charged_only and not is_charged(c):
                    continue
                constituents.append(c)

            if len(constituents) < 2:
                continue

            jet_pt = jet.pt()

            for i, ci in enumerate(constituents):
                for cj in constituents[i + 1:]:
                    dr = delta_r(ci, cj, self._use_pseudorapidity)
                    zi = ci.pt() / jet_pt
                    zj = cj.pt() / jet_pt
                    weight = (zi * zj) ** self._energy_weight

                    self._w_pairs += weight

                    # Pairs outside the binned range still count towards W_pairs, exactly as they would when
                    # histogramming stored pairs after the fact, so the normalization does not depend on the
                    # range the edges cover.
                    index = int(np.searchsorted(self._bin_edges, dr, side="right")) - 1
                    if 0 <= index < len(self._counts):
                        self._counts[index] += weight

    def scale(self, scale_factor: float) -> None:
        """
        The EEC is normalized by the total pair weight W_pairs and is therefore independent of the event
        normalization. Consequently, this method intentionally performs no operation.

        Parameters
        ----------
        scale_factor : float
            Multiplicative scale factor. Ignored.
        """
        pass

    @classmethod
    def finalize(cls, values: List[float], bin_edges: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Normalize the accumulated ΔR histogram into EEC(ΔR), by dividing by the total pair weight W_pairs and
        by each bin's width in ΔR.

        Parameters
        ----------
        values : List[float]
            The accumulated pair weight per ΔR bin.
        bin_edges : np.ndarray
            The bin edges the result is wanted on, which must be the ones the observable was filled with.
        metadata : Dict
            This observable's saved metadata, must contain ``W_PAIRS_KEY`` and ``BIN_EDGES_KEY``.

        Returns
        -------
        np.ndarray
            EEC(ΔR), one value per bin.

        Raises
        ------
        ValueError
            If the requested edges are not the ones the observable was filled on. Summing whole bins onto a
            coarser grid would be exact here as it is for the cross-section, but nothing asks for it -- the EEC
            is always read on the measurement's own edges -- so anything else is refused rather than
            approximated.
        """
        bin_edges = np.asarray(bin_edges, dtype=float)

        if cls.BIN_EDGES_KEY not in metadata:
            raise ValueError(
                "This EEC holds raw (ΔR, weight) pairs rather than a ΔR histogram, which no version of this "
                "observable still produces. Regenerate the simulation, which will accumulate it directly on "
                "the bin edges it is to be read on."
            )

        filled_edges = np.asarray(metadata[cls.BIN_EDGES_KEY], dtype=float)

        if filled_edges.shape != bin_edges.shape or not np.allclose(filled_edges, bin_edges):
            raise ValueError(
                f"This EEC was filled on {len(filled_edges) - 1} ΔR bins spanning "
                f"[{filled_edges[0]:.6g}, {filled_edges[-1]:.6g}], but finalize was asked for "
                f"{len(bin_edges) - 1} bins spanning [{bin_edges[0]:.6g}, {bin_edges[-1]:.6g}]. The pairs "
                f"needed to re-bin are no longer there; regenerate against the requested edges."
            )

        return np.asarray(values, dtype=float) / metadata[cls.W_PAIRS_KEY] / np.diff(bin_edges)

    @classmethod
    def normalization_denominator(cls, metadata: Dict, per_jet_normalized: bool) -> float:
        """
        The quantity this convention divides by to turn raw binned pair weight into an EEC.

        This is what a *flavor-summed* EEC must be recombined with. ``finalize`` divides by the sample's total
        pair weight and ``to_per_jet_normalized`` by its jet count, so for two flavors

            EEC_inclusive = (D_q * s_q + D_g * s_g) / (D_q + D_g)

        with D the value returned here -- and *not* with the flavor cross sections, which is only the same
        thing when both flavors carry equal pair weight per jet. They do not: gluon jets carry about 11% more
        at R = 0.4, so weighting by cross section mis-mixes the two shapes by a coherent tilt of a couple of
        percent across ΔR. In the per-jet convention D is the jet count, which *is* proportional to the cross
        section, so that convention is unaffected.

        Parameters
        ----------
        metadata : Dict
            This observable's saved metadata, must contain ``W_PAIRS_KEY`` and ``N_JETS_KEY``.
        per_jet_normalized : bool
            Whether the shape is per-jet normalized rather than normalized to unity.

        Returns
        -------
        float
            The flavor's EEC normalization denominator.
        """
        return float(metadata[cls.N_JETS_KEY] if per_jet_normalized else metadata[cls.W_PAIRS_KEY])

    @classmethod
    def to_per_jet_normalized(cls, shape: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Convert a unity-normalized EEC shape (as returned by ``finalize``) into the per-jet normalized
        Σ_EEC convention that measurements are typically published in -- i.e. divided by the number of jets
        rather than by the total pair weight, and summed over ordered constituent pairs rather than
        unordered ones (see ``ORDERED_PAIR_FACTOR``).

        Parameters
        ----------
        shape : np.ndarray
            Unity-normalized EEC(ΔR), one value per bin, as returned by ``finalize``.
        metadata : Dict
            This observable's saved metadata, must contain ``W_PAIRS_KEY`` and ``N_JETS_KEY``.

        Returns
        -------
        np.ndarray
            Σ_EEC(ΔR) per jet, one value per bin.
        """
        return cls.ORDERED_PAIR_FACTOR * shape * metadata[cls.W_PAIRS_KEY] / metadata[cls.N_JETS_KEY]

    def _get_complementary_metadata(self) -> dict:
        """
        Return complementary metadata.

        Returns
        -------
        complementary_metadata : dict
            Additional metadata.
        """
        return {
            self.W_PAIRS_KEY: self._w_pairs,
            self.N_JETS_KEY: self._n_jets,
            self.BIN_EDGES_KEY: self._bin_edges.tolist(),
            "energy_weight": self._energy_weight,
            "use_pseudorapidity": self._use_pseudorapidity,
        }

    def _get_binned_values(self) -> List[float]:
        """
        Return the pair weight accumulated in each ΔR bin so far.

        Returns
        -------
        List[float]
            The summed pair weight per ΔR bin.
        """
        return self._counts.tolist()

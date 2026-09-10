"""
    @file:              jet_finder.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 06/2026

    @Description:       This file defines the JetFinder class, a wrapper around FastJet for jet clustering and jet-level
                        selections on reconstructed particles.
"""

from typing import List, Optional

import fastjet as fj


class JetFinder:
    """
    A FastJet-based jet finder for clustering and applying jet-level selections.
    """
    def __init__(
        self,
        radius: float | int,
        pt_min: Optional[float | int] = None,
        pt_max: Optional[float | int] = None,
        max_abs_rapidity: Optional[float | int] = None,
        max_abs_pseudorapidity: Optional[float | int] = None,
        clustering_algorithm: int = fj.antikt_algorithm,
        recombination_scheme: int = fj.E_scheme,
        strategy: int = fj.Best
    ) -> None:
        """
        Constructor for JetFinder. Initializes the FastJet jet definition and selection cuts.

        Parameters
        ----------
        radius : float | int
            Jet radius parameter.
        pt_min : Optional[float | int]
            Minimum jet transverse momentum in GeV.
        pt_max : Optional[float | int]
            Maximum jet transverse momentum in GeV.
        max_abs_rapidity : Optional[float | int]
            Maximum absolute rapidity ``|y|`` for accepted jets (fiducial cut).
        max_abs_pseudorapidity : Optional[float | int]
            Maximum absolute pseudorapidity ``|eta|`` for accepted jets (fiducial cut).
        clustering_algorithm : int
            FastJet clustering algorithm (e.g. fj.antikt_algorithm).
        recombination_scheme : int
            FastJet recombination scheme (e.g. fj.E_scheme).
        strategy : int
            FastJet clustering strategy (e.g. fj.Best).
        """
        self._jet_definition = fj.JetDefinition(clustering_algorithm, radius, recombination_scheme, strategy)

        selector = fj.SelectorIdentity()
        if pt_min:
            selector &= fj.SelectorPtMin(pt_min)
        if pt_max:
            selector &= fj.SelectorPtMax(pt_max)
        if max_abs_rapidity:
            selector &= fj.SelectorAbsRapMax(max_abs_rapidity)
        if max_abs_pseudorapidity:
            selector &= fj.SelectorAbsEtaMax(max_abs_pseudorapidity)

        self._selector = selector

    def get_cluster(
        self,
        particles: List[fj.PseudoJet]
    ) -> fj.ClusterSequence:
        """
        Cluster particles into a FastJet ClusterSequence.

        Parameters
        ----------
        particles : List[fj.PseudoJet]
            Input particles as FastJet PseudoJets.

        Returns
        -------
        cluster_sequence : fj.ClusterSequence
            FastJet ClusterSequence.
        """
        return fj.ClusterSequence(
            particles,
            self._jet_definition
        )

    def find_jets(
        self,
        cluster_sequence: fj.ClusterSequence
    ) -> List[fj.PseudoJet]:
        """
        Return jets passing the kinematic selections.

        Parameters
        ----------
        cluster_sequence : fj.ClusterSequence
            FastJet ClusterSequence.

        Returns
        -------
        jets : List[fj.PseudoJet]
            Reconstructed jets sorted by transverse momentum (highest first),
            after applying pT and rapidity selections.
        """
        inclusive_jets = fj.sorted_by_pt(cluster_sequence.inclusive_jets())

        return self._selector(inclusive_jets)

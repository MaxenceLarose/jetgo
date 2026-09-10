"""
    @file:              descendancy_tracing.py
    @Author:            Maxence Larose

    @Creation Date:     07/2026
    @Last modification: 07/2026

    @Description:       This file defines the DescendancyTracing tagger, which tags a jet via a pT-weighted
                        average vote cast by each hard-scattering parton, or one of its descendants, that reaches
                        geometrically close to the jet axis. Jets left untagged by that search can optionally fall
                        back to every quark/gluon daughter of the relevant beam-remnant parton (found by tracing the
                        jet's own constituents backward), letting them all vote the same way, to catch jets that
                        genuinely originate from the beam remnant rather than from the two outgoing
                        hard-scattering partons.
"""

from enum import StrEnum
from typing import Dict, List, Optional, Set, TYPE_CHECKING, Tuple

import fastjet as fj

from .base import JetFlavorTagger
from .flavor import JetFlavor
from .identifier import JetFlavorTaggerIdentifier
from ..kinematics import delta_r, pythia_index

if TYPE_CHECKING:
    from .._pythia import pythia8


class FlavorDecisionRule(StrEnum):
    """
    How ``DescendancyTracing`` picks the winning flavor out of a set of pT-weighted votes (see
    ``DescendancyTracing._decide_flavor``).
    """

    AVERAGE_PT = "average_pt"
    CLOSEST_PT_TO_JET = "closest_pt_to_jet"


class DescendancyTracing(JetFlavorTagger):
    """
    Tags a jet with a pT-weighted average vote cast by each hard-scattering parton, or one of its descendants, that
    reaches geometrically close to the jet axis.

    For each hard-scattering (``|status|`` == 23) parton, the parton itself and its shower are walked forward generation
    by generation, starting with the parton itself (generation 0), then its daughters (generation 1), then daughters
    of daughters, and so on. At each generation, every particle in it casts a vote (weighted by its pT) for its
    parton's flavor, but only if it satisfies both:

    - It is within ``delta_r_max`` of the jet axis;
    - It is one of the jet's constituents, or an ancestor of one.

    Establishing that second condition is a two-step process, and neither step ever uses a mother pointer:

    1. Forward check (``_valid_descendants_of``): does the hard parton have *any* descendant that is a jet
       constituent at all, via ``daughterListRecursive()``? If not, this parton contributes nothing, no matter how
       far the search below expands.
    2. Forward ancestor check (``_is_ancestor_of_seed_constituent``): every candidate the generation-by-generation
       search below actually visits -- not the hard parton's entire shower up front, since with the default
       ``min_generations`` the search almost always settles well before reaching final-state hadronization, so most
       of that shower is never visited at all -- is checked, one at a time, the same way: does *its own*
       ``daughterListRecursive()`` reach one of the constituents confirmed in step 1? If so, it is recognized as
       "on the path" and may cast a vote, even though it isn't a jet constituent itself. This is needed because the
       search usually finds a candidate long before it reaches the constituent itself.

       Mother pointers (``mother1()``/``mother2()``) are deliberately not used for either step. PYTHIA8's
       color reconnection can rearrange which partons end up color-connected into the same hadronizing
       string, so a hadron's recorded mother1()/mother2() can point somewhere entirely
       disconnected from its true shower-emission history, even when a forward path from the hard parton to that
       hadron genuinely exists. Both steps here rely only on ``daughterListRecursive()``, which is unaffected by
       this and remains exhaustive.

    At least ``min_generations`` generations are always explored for every hard parton (not just until the first
    parton to find a vote), so that partons whose shower happens to resolve close to the jet earlier don't cut the
    search short for a parton that would only be identified deeper in its own shower. If no parton has collected any
    vote after ``min_generations`` generations, the search keeps expanding, for every parton simultaneously, one
    generation at a time, until at least one vote is found or every parton's shower is fully exhausted.

    The jet is tagged according to ``decision_rule`` (see ``_decide_flavor``): by default, whichever flavor has
    the higher average pT among its own collected votes (not the sum, so a flavor with fewer but harder matching
    particles can outrank one with more but softer ones). If neither flavor collects any vote, the jet is left
    untagged by this search.

    If ``handle_untagged_jets`` is True (the default), a jet still untagged at that point gets one more chance: some
    jets genuinely originate from a beam-remnant parton (spawned to conserve color/quantum numbers once the actual
    hard-scattering parton is removed from the incoming proton) rather than from either of the two outgoing
    hard-scattering partons, so no ``|status|`` == 23 parton was ever going to claim them. For such a jet, every
    constituent is traced backward through ``mother1()`` to its earliest quark/gluon ancestor (see
    ``_trace_to_earliest_ancestor``), and that ancestor's own mother -- typically the incoming beam proton itself --
    is used to enumerate every one of its quark/gluon daughters (see ``_find_fallback_hard_partons``).

    Rather than picking a single substitute parton, every one of these candidates is run through the exact same
    generation-by-generation forward search as a real hard parton, so the same pT-weighted-average vote decides the
    flavor -- this mirrors how the primary search above never commits to a single hard parton in advance either; it
    considers every ``|status|`` == 23 parton in the event and lets them all vote. Committing to one substitute would be
    unprincipled here anyway: these beam-remnant partons are typically pieces of a single jointly-fragmenting color
    system, so more than one of them can legitimately claim to have contributed to the same jet. If no candidate
    collects any vote, or no constituent's lineage passes through a quark or a gluon at all, the jet remains
    untagged.
    """

    MAX_ANCESTRY_STEPS = 500

    def __init__(
        self,
        delta_r_max: float | int,
        min_generations: int = 5,
        max_generations: int = 100,
        use_pseudorapidity: bool = False,
        handle_untagged_jets: bool = True,
        decision_rule: FlavorDecisionRule | str = FlavorDecisionRule.AVERAGE_PT
    ) -> None:
        """
        Constructor for DescendancyTracing.

        Parameters
        ----------
        delta_r_max : float | int
            Maximum angular distance ΔR between the jet axis and a shower descendant for that descendant to be
            considered a match. Should typically be set equal to the jet radius.
        min_generations : int, default=5
            Minimum number of generations (daughters, daughters of daughters, ...) to walk forward from each hard
            parton before considering the search complete.
        max_generations : int, default=100
            Safety cap on how many generations to walk forward if no vote has been found after
            ``min_generations``, in case a shower's every branch keeps producing daughters without ever
            terminating in a match.
        use_pseudorapidity : bool, default=False
            If True, ΔR is computed using the pseudorapidity η instead of the (true, mass-dependent) rapidity y, i.e.
            ΔR = sqrt((Δη)² + (Δφ)²) instead of ΔR = sqrt((Δy)² + (Δφ)²).
        handle_untagged_jets : bool, default=True
            If True, a jet left untagged by the normal hard-parton search gets a second attempt with candidate
            beam-remnant partons found from the jet's own constituents (see the class docstring). If False, such a
            jet is simply left untagged.
        decision_rule : FlavorDecisionRule or str, default=FlavorDecisionRule.AVERAGE_PT
            How to pick the winning flavor out of a set of votes (see ``_decide_flavor``). Also accepts the
            matching raw string (e.g. ``"closest_pt_to_jet"``); an invalid string raises ``ValueError``.
        """
        self._delta_r_max = delta_r_max
        self._min_generations = min_generations
        self._max_generations = max_generations
        self._use_pseudorapidity = use_pseudorapidity
        self._handle_untagged_jets = handle_untagged_jets
        self._decision_rule = FlavorDecisionRule(decision_rule)
        self._fallback_call_count = 0

    def get_diagnostics(self) -> dict:
        """
        Return how many times the fallback (see ``handle_untagged_jets``) was invoked, i.e. how many jets the
        normal hard-parton search alone left untagged.

        Returns
        -------
        dict
            ``{"fallback_call_count": <int>}``.
        """
        return {"fallback_call_count": self._fallback_call_count}

    @property
    def identifier(self) -> JetFlavorTaggerIdentifier:
        """
        Identifier of the tagging strategy.

        Returns
        -------
        identifier : JetFlavorTaggerIdentifier
            Identifier of the tagging strategy.
        """
        return JetFlavorTaggerIdentifier.DESCENDANCY_TRACING

    # ----------------------------------------------------------------------------------------------------------- #
    # Step 1 + 2: deciding which particles in a hard parton's shower are valid votes (constituent or ancestor of one)
    # ----------------------------------------------------------------------------------------------------------- #

    @staticmethod
    def _hard_parton_descendants(event: "pythia8.Event", hard_parton_index: int) -> Set[int]:
        """
        Return every particle in ``hard_parton_index``'s shower, at any depth, plus the hard parton itself.

        Parameters
        ----------
        event : pythia8.Event
            Full Pythia8 event record.
        hard_parton_index : int
            Event-record index of the hard-scattering parton.

        Returns
        -------
        Set[int]
            ``{hard_parton_index}`` union its full recursive daughter set.
        """
        descendants = {hard_parton_index}
        descendants.update(event[hard_parton_index].daughterListRecursive())
        return descendants

    @staticmethod
    def _reachable_constituents(jet: fj.PseudoJet, descendants: Set[int]) -> Set[int]:
        """
        Step 1 (forward check): which of the jet's constituents are reachable descendants of the hard parton?

        Parameters
        ----------
        jet : fj.PseudoJet
            Reconstructed jet whose constituents are checked.
        descendants : Set[int]
            The hard parton's own descendant set (see ``_hard_parton_descendants``).

        Returns
        -------
        Set[int]
            Constituent indices that are also in ``descendants``. Empty if the hard parton has no descendant
            constituent at all, in which case it cannot contribute any vote for this jet.
        """
        constituent_indices = {pythia_index(c) for c in jet.constituents()}
        return constituent_indices & descendants

    @staticmethod
    def _is_ancestor_of_seed_constituent(
        event: "pythia8.Event",
        candidate_index: int,
        seed_constituents: Set[int]
    ) -> bool:
        """
        Step 2 (forward ancestor check): is ``candidate_index`` itself one of ``seed_constituents``, or an
        ancestor of one -- determined entirely via forward daughter reachability (``daughterListRecursive()``),
        never via mother pointers (see the class docstring for why).

        This is checked lazily, one candidate at a time, only for whichever particle the generation-by-generation
        search in ``_collect_votes`` actually visits -- not precomputed for the hard parton's entire shower. Most
        of that shower (everything past wherever the search settles, usually still deep within the parton shower,
        well short of final-state hadronization) is never visited at all, since ``min_generations`` typically stops
        the search long before reaching it, so validating it up front would mean checking far more particles than
        the search ever needs.

        Parameters
        ----------
        event : pythia8.Event
            Full Pythia8 event record.
        candidate_index : int
            Event-record index of the particle being considered as a vote candidate.
        seed_constituents : Set[int]
            Constituents already confirmed to descend from the hard parton (see ``_reachable_constituents``).

        Returns
        -------
        bool
            True if ``candidate_index`` is a seed constituent, or an ancestor of one.
        """
        if candidate_index in seed_constituents:
            return True

        candidate_descendants = set(event[candidate_index].daughterListRecursive())
        return bool(candidate_descendants & seed_constituents)

    def _valid_descendants_of(
        self,
        jet: fj.PseudoJet,
        event: "pythia8.Event",
        hard_parton_index: int
    ) -> Set[int]:
        """
        Return the jet constituents reachable from ``hard_parton_index`` (see ``_reachable_constituents``), i.e.
        whether this hard parton contributes to this jet at all. Empty if it has no descendant constituent, in
        which case it cannot contribute any vote for this jet, no matter how far ``_collect_votes`` expands.

        Parameters
        ----------
        jet : fj.PseudoJet
            Reconstructed jet whose constituents are checked.
        event : pythia8.Event
            Full Pythia8 event record.
        hard_parton_index : int
            Event-record index of the hard-scattering parton whose shower is being validated.

        Returns
        -------
        Set[int]
            The jet's own constituent indices that are confirmed descendants of ``hard_parton_index``.
        """
        descendants = self._hard_parton_descendants(event, hard_parton_index)
        return self._reachable_constituents(jet, descendants)

    # ----------------------------------------------------------------------------------------------------------- #
    # The generation-by-generation vote collection and final decision
    # ----------------------------------------------------------------------------------------------------------- #

    def _find_hard_partons(self, event: "pythia8.Event") -> List[Tuple[int, JetFlavor]]:
        """
        Find every hard-scattering (``|status|`` == 23) parton in the event that maps to a quark or a gluon flavor.

        Parameters
        ----------
        event : pythia8.Event
            Full Pythia8 event record.

        Returns
        -------
        List[Tuple[int, JetFlavor]]
            (event-record index, flavor) for each qualifying hard parton.
        """
        hard_partons: List[Tuple[int, JetFlavor]] = []

        for index, particle in enumerate(event):
            if abs(particle.status()) != self.HARD_PROCESS_STATUS:
                continue

            flavor = self._get_flavor_from_pdg_id(particle.id())
            if flavor != JetFlavor.UNTAGGED:
                hard_partons.append((index, flavor))

        return hard_partons

    def _cast_votes_in_frontier(
        self,
        jet: fj.PseudoJet,
        event: "pythia8.Event",
        frontier: Set[int],
        seed_constituents: Set[int]
    ) -> List[float]:
        """
        Among one generation's worth of candidates, return the pT of every one that is within ``delta_r_max`` of
        the jet axis and a jet constituent or an ancestor of one (see ``_is_ancestor_of_seed_constituent``).

        Parameters
        ----------
        jet : fj.PseudoJet
            Reconstructed jet being tagged.
        event : pythia8.Event
            Full Pythia8 event record.
        frontier : Set[int]
            This generation's candidate indices for one hard parton.
        seed_constituents : Set[int]
            That hard parton's reachable jet constituents (see ``_valid_descendants_of``).

        Returns
        -------
        List[float]
            pT of every qualifying candidate in ``frontier``.
        """
        votes: List[float] = []

        for descendant_index in frontier:
            if not self._is_ancestor_of_seed_constituent(event, descendant_index, seed_constituents):
                continue

            descendant = event[descendant_index]
            descendant_as_pseudo_jet = fj.PseudoJet(descendant.px(), descendant.py(), descendant.pz(), descendant.e())

            if delta_r(jet, descendant_as_pseudo_jet, self._use_pseudorapidity) < self._delta_r_max:
                votes.append(descendant.pT())

        return votes

    def _collect_votes(
        self,
        jet: fj.PseudoJet,
        event: "pythia8.Event",
        hard_partons: List[Tuple[int, JetFlavor]]
    ) -> Dict[JetFlavor, List[float]]:
        """
        Walk every hard parton's shower forward, generation by generation, collecting pT-weighted votes per flavor.
        Generation 0 is the hard parton itself, so it may cast a vote too, not just its descendants.

        Parameters
        ----------
        jet : fj.PseudoJet
            Reconstructed jet to tag.
        event : pythia8.Event
            Full Pythia8 event record.
        hard_partons : List[Tuple[int, JetFlavor]]
            (event-record index, flavor) for each hard parton, as returned by ``_find_hard_partons``.

        Returns
        -------
        Dict[JetFlavor, List[float]]
            pT of every qualifying vote, keyed by flavor.
        """
        seed_constituents_list = [self._valid_descendants_of(jet, event, index) for index, _ in hard_partons]
        frontiers = [{index} for index, _ in hard_partons]
        votes: Dict[JetFlavor, List[float]] = {JetFlavor.QUARK: [], JetFlavor.GLUON: []}

        # Generation 0: the hard parton itself may also cast a vote.
        for i, (_, flavor) in enumerate(hard_partons):
            votes[flavor].extend(self._cast_votes_in_frontier(jet, event, frontiers[i], seed_constituents_list[i]))

        for generation in range(1, self._max_generations + 1):
            any_frontier_nonempty = False

            for i, (_, flavor) in enumerate(hard_partons):
                next_frontier: Set[int] = set()
                for index in frontiers[i]:
                    next_frontier.update(event[index].daughterList())
                frontiers[i] = next_frontier

                if next_frontier:
                    any_frontier_nonempty = True

                votes[flavor].extend(self._cast_votes_in_frontier(jet, event, next_frontier, seed_constituents_list[i]))

            has_votes = bool(votes[JetFlavor.QUARK] or votes[JetFlavor.GLUON])

            if generation >= self._min_generations and (has_votes or not any_frontier_nonempty):
                break

        return votes

    def _decide_flavor(self, jet: fj.PseudoJet, votes: Dict[JetFlavor, List[float]]) -> JetFlavor:
        """
        Decide the winning flavor from the collected votes, according to ``decision_rule``:

        - ``FlavorDecisionRule.AVERAGE_PT`` (the default): whichever flavor has the higher average pT among its
          own votes (not the sum, so a flavor with fewer but harder matching particles can outrank one with more
          but softer ones).
        - ``FlavorDecisionRule.CLOSEST_PT_TO_JET``: every vote, both flavors pooled together, is compared to the
          jet's own pT, and the flavor of whichever single vote is closest wins -- a "leading particle" style
          decision using only the one vote that best matches the jet's own momentum.

        Parameters
        ----------
        jet : fj.PseudoJet
            Reconstructed jet being tagged. Only used by ``FlavorDecisionRule.CLOSEST_PT_TO_JET``.
        votes : Dict[JetFlavor, List[float]]
            pT of every qualifying vote, keyed by flavor (see ``_collect_votes``).

        Returns
        -------
        JetFlavor
            The winning flavor, or ``JetFlavor.UNTAGGED`` if neither flavor collected any vote.
        """
        quark_votes = votes[JetFlavor.QUARK]
        gluon_votes = votes[JetFlavor.GLUON]

        if not quark_votes and not gluon_votes:
            return JetFlavor.UNTAGGED

        if self._decision_rule == FlavorDecisionRule.CLOSEST_PT_TO_JET:
            all_votes = [(pt, JetFlavor.QUARK) for pt in quark_votes] + [(pt, JetFlavor.GLUON) for pt in gluon_votes]
            _, flavor = min(all_votes, key=lambda vote: abs(vote[0] - jet.pt()))
            return flavor

        quark_average = sum(quark_votes) / len(quark_votes) if quark_votes else -1.0
        gluon_average = sum(gluon_votes) / len(gluon_votes) if gluon_votes else -1.0

        return JetFlavor.QUARK if quark_average > gluon_average else JetFlavor.GLUON

    # ----------------------------------------------------------------------------------------------------------- #
    # Fallback for jets left untagged by the normal hard-parton search (see ``handle_untagged_jets``)
    # ----------------------------------------------------------------------------------------------------------- #

    def _trace_to_earliest_ancestor(
        self,
        event: "pythia8.Event",
        start_index: int,
        max_steps: int = MAX_ANCESTRY_STEPS
    ) -> Optional[int]:
        """
        Walk the mother lineage of ``event[start_index]`` back through ``mother1()`` as far as it goes, with
        no stopping condition on status. It does not stop at ``|status|`` == 23 or at beam/ISR/MPI lineage,
        since the whole point is to reach past the two outgoing hard-scattering partons to whatever
        initiated this branch of the event in the first place.

        Walking indiscriminately all the way to ``mother1() <= 0`` would overshoot past the useful parton-level
        history and land on the incoming beam proton itself, which is not a quark or a gluon. So instead, the
        *last* ancestor along the way that is still a quark or a gluon is what gets returned -- typically an
        initial-state-radiation parton -- rather than the true, non-partonic root of the whole lineage.

        Parameters
        ----------
        event : pythia8.Event
            Full Pythia8 event record.
        start_index : int
            Event-record index of the particle to start tracing from.
        max_steps : int
            Maximum number of mother-lineage steps to walk before giving up.

        Returns
        -------
        Optional[int]
            The earliest quark/gluon ancestor found along the lineage, or None if the lineage never passes through
            one (e.g. it leads directly to a beam proton, a lepton, ...).
        """
        current = start_index
        earliest_parton: Optional[int] = None

        for _ in range(max_steps):
            if self._get_flavor_from_pdg_id(event[current].id()) != JetFlavor.UNTAGGED:
                earliest_parton = current

            mother_1 = event[current].mother1()

            if mother_1 <= 0 or mother_1 == current:
                break

            current = mother_1

        return earliest_parton

    def _find_fallback_hard_partons(
        self,
        jet: fj.PseudoJet,
        event: "pythia8.Event"
    ) -> List[Tuple[int, JetFlavor]]:
        """
        Find substitute "hard partons" for a jet left untagged by every ``|status|`` == 23 parton, by tracing the jet's
        own constituents backward to the earliest quark/gluon ancestor's own mother -- typically the incoming beam
        proton, once the actual hard-scattering parton has been removed from it -- and taking every one of that
        mother's quark/gluon daughters as a candidate.

        No single candidate is picked here: every one of them is returned, to be run through the same
        generation-by-generation forward search as a real hard parton (see the class docstring for why).

        Parameters
        ----------
        jet : fj.PseudoJet
            Reconstructed jet with no vote from any ``|status|`` == 23 hard parton.
        event : pythia8.Event
            Full Pythia8 event record.

        Returns
        -------
        List[Tuple[int, JetFlavor]]
            (event-record index, flavor) for every quark/gluon daughter of every relevant beam-remnant mother found
            from the jet's constituents. Empty if no constituent's lineage passes through a quark or a gluon at all.
        """
        mother_indices: Set[int] = set()

        for constituent in jet.constituents():
            constituent_index = pythia_index(constituent)
            if not (0 < constituent_index < event.size()):
                continue

            earliest_parton = self._trace_to_earliest_ancestor(event, constituent_index)
            if earliest_parton is None:
                continue

            mother_index = event[earliest_parton].mother1()
            if mother_index > 0:
                mother_indices.add(mother_index)

        fallback_hard_partons: List[Tuple[int, JetFlavor]] = []
        seen_indices: Set[int] = set()

        for mother_index in mother_indices:
            for daughter_index in event[mother_index].daughterList():
                if daughter_index in seen_indices:
                    continue
                seen_indices.add(daughter_index)

                flavor = self._get_flavor_from_pdg_id(event[daughter_index].id())
                if flavor != JetFlavor.UNTAGGED:
                    fallback_hard_partons.append((daughter_index, flavor))

        return fallback_hard_partons

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
            yielded by the generator, since it needs to still contain the original hard-scattering partons and their
            full shower history.

        Returns
        -------
        flavor : JetFlavor
            ``JetFlavor.QUARK`` or ``JetFlavor.GLUON`` if the jet was successfully tagged, ``JetFlavor.UNTAGGED``
            otherwise.
        """
        hard_partons = self._find_hard_partons(event)
        flavor = JetFlavor.UNTAGGED

        if hard_partons:
            votes = self._collect_votes(jet, event, hard_partons)
            flavor = self._decide_flavor(jet, votes)

        if flavor == JetFlavor.UNTAGGED and self._handle_untagged_jets:
            self._fallback_call_count += 1
            fallback_hard_partons = self._find_fallback_hard_partons(jet, event)
            if fallback_hard_partons:
                votes = self._collect_votes(jet, event, fallback_hard_partons)
                flavor = self._decide_flavor(jet, votes)

        return flavor

"""
    @file:              simulator.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 07/2026

    @Description:       This file defines the Simulator class, which orchestrates the full jet analysis workflow.
                        It generates events, selects particles, reconstructs jets, evaluates observables, and saves
                        the resulting histograms to a single JSON file.
"""

import copy
import json
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from .jet_finder import JetFinder
from .particle_selector import ParticleSelector
from .event_generator import EventGenerator
from .simulation_summary import SimulationSummary, TaggerPair

from ..observables.base import Observable
from ..taggers.base import JetFlavorTagger
from ..taggers.flavor import JetFlavor

T = TypeVar("T")

# For each tagger, the (quark-tagged, gluon-tagged) copy of every observable.
TaggedObservables = Dict[JetFlavorTagger, Dict[Observable, Tuple[Observable, Observable]]]


class Simulator:
    """
    High-level driver for a jet physics analysis workflow.

    This class orchestrates the full event-level analysis pipeline: event generation, particle selection, jet
    reconstruction, and observable evaluation.

    For each event, particles are selected according some acceptance criteria, clustered into jets, and passed to the
    observable for filling the histogram.
    """

    def __init__(
        self,
        event_generator: EventGenerator,
        jet_finder: JetFinder,
        particle_selector: ParticleSelector
    ) -> None:
        """
        Constructor. It initializes the event generator, jet finder, and particle selector.

        Parameters
        ----------
        event_generator : EventGenerator
            Event generator used to generate collision events.
        jet_finder : JetFinder
            Jet reconstruction algorithm used to cluster particles into jets.
        particle_selector : ParticleSelector
            Particle selection algorithm used to select particles.
        """
        self._generator = event_generator
        self._jet_finder = jet_finder
        self._particle_selector = particle_selector

    def simulate(
            self,
            n_events: int,
            observables: Observable | Sequence[Observable],
            output_path: str | Path,
            jet_flavor_taggers: Optional[JetFlavorTagger | Sequence[JetFlavorTagger]] = None,
            verbose: bool = False
    ) -> None:
        """
        Run the full event loop for one or more observables, optionally split by jet flavor.

        Each event is processed sequentially: generation → particle selection → jet finding → observable histogram
        filling. Every observable in ``observables`` is filled with all reconstructed jets, regardless of flavor.

        If ``jet_flavor_taggers`` is given, each tagger additionally produces a quark-tagged and a gluon-tagged copy of
        every observable (independent deep copies: same class, same construction parameters, empty histograms),
        filled only with the jets that tagger labels as a quark or a gluon respectively. Jets a tagger leaves
        untagged are excluded from both of its copies, but still contribute to the inclusive histogram.

        All results are written to a single JSON file: run metadata (number of events, number of jets, and, for
        every tagger, its number of quark-tagged/gluon-tagged/untagged jets, plus, for every pair of taggers, how
        often they agree on flavor among jets both of them tag), plus the inclusive histogram of every observable
        and the quark/gluon histograms for every (observable, tagger) pair, keyed by ``ObservableIdentifier`` and
        ``JetFlavorTaggerIdentifier`` respectively.

        Parameters
        ----------
        n_events : int
            Number of events to generate.
        observables : Observable or Sequence[Observable]
            One observable, or a sequence of observables, whose internal histograms are filled event-by-event with
            all jets. Each observable must have a distinct ``identifier``, since it is used as that observable's key
            in the output file.
        output_path : str or Path
            Output JSON file where every result is stored. Created or overwritten.
        jet_flavor_taggers : Optional[JetFlavorTagger or Sequence[JetFlavorTagger]]
            One tagger, or a sequence of taggers, used to label each jet as a quark or a gluon. Each tagger must have
            a distinct ``identifier``. If None (default), only inclusive histograms are produced.
        verbose : bool
            Verbose mode.
        """
        observables = self._normalize_observables(observables)
        jet_flavor_taggers = self._normalize_taggers(jet_flavor_taggers)

        tagged_observables = self._build_tagged_observables(observables, jet_flavor_taggers)

        simulation_summary = self._run_event_loop(n_events, observables, tagged_observables, verbose)
        self._scale_all(observables, tagged_observables, self._generator.scale_factor)

        self._save_results(observables, tagged_observables, output_path, simulation_summary)

        if verbose:
            self._generator.print_statistics()
            simulation_summary.print_statistics()

    def _normalize_observables(self, observables: Observable | Sequence[Observable]) -> List[Observable]:
        """
        Normalize a single observable or a sequence of observables into a list, and check that every one has a
        distinct ``identifier``.

        Parameters
        ----------
        observables : Observable or Sequence[Observable]
            One observable, or a sequence of observables.

        Returns
        -------
        List[Observable]
            The normalized list.
        """
        if isinstance(observables, Observable):
            observables = [observables]

        self._check_unique_identifiers(observables, key=lambda o: o.identifier, kind="observable")

        return list(observables)

    def _normalize_taggers(
        self,
        jet_flavor_taggers: Optional[JetFlavorTagger | Sequence[JetFlavorTagger]]
    ) -> List[JetFlavorTagger]:
        """
        Normalize a single tagger, a sequence of taggers, or None into a list, and check that every tagger has a
        distinct ``identifier``.

        Parameters
        ----------
        jet_flavor_taggers : Optional[JetFlavorTagger or Sequence[JetFlavorTagger]]
            One tagger, a sequence of taggers, or None.

        Returns
        -------
        List[JetFlavorTagger]
            The normalized list. Empty if ``jet_flavor_taggers`` was None.
        """
        if isinstance(jet_flavor_taggers, JetFlavorTagger):
            jet_flavor_taggers = [jet_flavor_taggers]

        jet_flavor_taggers = list(jet_flavor_taggers or [])

        self._check_unique_identifiers(jet_flavor_taggers, key=lambda t: t.identifier, kind="flavor tagger")

        return jet_flavor_taggers

    def _build_tagged_observables(
        self,
        observables: Sequence[Observable],
        jet_flavor_taggers: Sequence[JetFlavorTagger]
    ) -> TaggedObservables:
        """
        For each tagger, build a (quark-tagged, gluon-tagged) deep copy of every observable, with empty histograms.

        Parameters
        ----------
        observables : Sequence[Observable]
            Observables to copy, one pair of copies per tagger.
        jet_flavor_taggers : Sequence[JetFlavorTagger]
            Taggers to build copies for.

        Returns
        -------
        TaggedObservables
            For each tagger, the (quark, gluon) observable copies for each observable.
        """
        return {
            tagger: {
                observable: (copy.deepcopy(observable), copy.deepcopy(observable)) for observable in observables
            }
            for tagger in jet_flavor_taggers
        }

    def _run_event_loop(
        self,
        n_events: int,
        observables: Sequence[Observable],
        tagged_observables: TaggedObservables,
        verbose: bool
    ) -> SimulationSummary:
        """
        Generate ``n_events`` events and fill every observable (inclusive and flavor-tagged) event by event, while
        accumulating the jet counts and pairwise flavor agreement saved as run metadata.

        Every jet is tagged exactly once per tagger (rather than delegating to a per-tagger helper that would tag it
        independently), so that, in the same pass, every pair of taggers can be compared on the same jet.

        Parameters
        ----------
        n_events : int
            Number of events to generate.
        observables : Sequence[Observable]
            Observables filled with every reconstructed jet, regardless of flavor.
        tagged_observables : TaggedObservables
            For each tagger, the (quark, gluon) observable copies to fill with flavor-split jets.
        verbose : bool
            If True, print progress every 100 events.

        Returns
        -------
        SimulationSummary
            Number of events and jets processed, and, for every tagger, its counts per flavor and its pairwise
            agreement with every other tagger.
        """
        taggers = list(tagged_observables.keys())
        tagger_pairs: List[TaggerPair] = list(combinations(taggers, 2))

        n_jets = 0
        tagger_counts: Dict[JetFlavorTagger, Dict[JetFlavor, int]] = {
            tagger: {JetFlavor.QUARK: 0, JetFlavor.GLUON: 0, JetFlavor.UNTAGGED: 0} for tagger in taggers
        }
        pairwise_agreement: Dict[TaggerPair, Dict[str, int]] = {
            pair: {"both_tagged": 0, "agree": 0} for pair in tagger_pairs
        }

        for event_idx, event in enumerate(self._generator(n_events), start=1):
            if verbose and event_idx % 100 == 0:
                print(f"Generating event {event_idx}")

            particles = self._particle_selector.select(event)
            cluster = self._jet_finder.get_cluster(particles)
            jets = self._jet_finder.find_jets(cluster)
            n_jets += len(jets)

            for observable in observables:
                observable.fill(jets=jets, cluster=cluster)

            jets_by_flavor = {tagger: {JetFlavor.QUARK: [], JetFlavor.GLUON: []} for tagger in taggers}

            for jet in jets:
                flavors = {tagger: tagger.tag(jet, event) for tagger in taggers}

                for tagger, flavor in flavors.items():
                    tagger_counts[tagger][flavor] += 1

                    if flavor != JetFlavor.UNTAGGED:
                        jets_by_flavor[tagger][flavor].append(jet)

                for tagger_a, tagger_b in tagger_pairs:
                    flavor_a, flavor_b = flavors[tagger_a], flavors[tagger_b]

                    if flavor_a != JetFlavor.UNTAGGED and flavor_b != JetFlavor.UNTAGGED:
                        pair_counts = pairwise_agreement[(tagger_a, tagger_b)]
                        pair_counts["both_tagged"] += 1

                        if flavor_a == flavor_b:
                            pair_counts["agree"] += 1

            for tagger, per_observable in tagged_observables.items():
                quark_jets = jets_by_flavor[tagger][JetFlavor.QUARK]
                gluon_jets = jets_by_flavor[tagger][JetFlavor.GLUON]

                for quark_observable, gluon_observable in per_observable.values():
                    quark_observable.fill(jets=quark_jets, cluster=cluster)
                    gluon_observable.fill(jets=gluon_jets, cluster=cluster)

        return SimulationSummary(
            n_events=n_events,
            n_jets=n_jets,
            tagger_counts=tagger_counts,
            pairwise_agreement=pairwise_agreement
        )

    def _scale_all(
        self,
        observables: Sequence[Observable],
        tagged_observables: TaggedObservables,
        scale_factor: float
    ) -> None:
        """
        Apply the event-normalization scale factor to every observable's histogram, inclusive and flavor-tagged.

        Parameters
        ----------
        observables : Sequence[Observable]
            Observables to scale.
        tagged_observables : TaggedObservables
            For each tagger, the (quark, gluon) observable copies to scale.
        scale_factor : float
            Factor by which every histogram is multiplied.
        """
        for observable in observables:
            observable.scale(scale_factor=scale_factor)

        for per_observable in tagged_observables.values():
            for quark_observable, gluon_observable in per_observable.values():
                quark_observable.scale(scale_factor=scale_factor)
                gluon_observable.scale(scale_factor=scale_factor)

    def _check_unique_identifiers(
        self,
        items: Sequence[T],
        key: Callable[[T], object],
        kind: str
    ) -> None:
        """
        Raise a ``ValueError`` if two or more items share the same identifier, since identifiers are used as keys in
        the output file and would otherwise silently overwrite one another.

        Parameters
        ----------
        items : Sequence[T]
            Items to check, e.g. observables or flavor taggers.
        key : Callable[[T], object]
            Function extracting the identifier from an item.
        kind : str
            Human-readable name for the item kind, used in the error message.
        """
        identifiers = [key(item) for item in items]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Duplicate {kind} identifiers: {identifiers}. Each {kind} must be unique per run.")

    def _save_results(
        self,
        observables: Sequence[Observable],
        tagged_observables: TaggedObservables,
        output_path: str | Path,
        simulation_summary: SimulationSummary
    ) -> None:
        """
        Write the run metadata and every observable's inclusive and (if any) flavor-tagged histograms to a single
        JSON file.

        Parameters
        ----------
        observables : Sequence[Observable]
            Observables whose inclusive histograms are saved.
        tagged_observables : TaggedObservables
            For each tagger, the (quark, gluon) observable copies for each observable.
        output_path : str or Path
            Destination ``.json`` file path. The file is created or overwritten.
        simulation_summary : SimulationSummary
            Statistics accumulated over the run (see ``_run_event_loop``), saved under the ``"metadata"`` key.
        """
        results: Dict[str, dict] = {"metadata": simulation_summary.to_dict()}

        results.update({
            observable.identifier: {
                "histograms": {"inclusive": observable.to_dict()},
            }
            for observable in observables
        })

        for tagger, per_observable in tagged_observables.items():
            for observable, (quark_observable, gluon_observable) in per_observable.items():
                results[observable.identifier]["histograms"][tagger.identifier] = {
                    JetFlavor.QUARK: quark_observable.to_dict(),
                    JetFlavor.GLUON: gluon_observable.to_dict(),
                }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

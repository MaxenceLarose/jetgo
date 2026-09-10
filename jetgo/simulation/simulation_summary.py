"""
    @file:              simulation_summary.py
    @Author:            Maxence Larose

    @Creation Date:     07/2026
    @Last modification: 07/2026

    @Description:       This file defines the SimulationSummary dataclass, which accumulates jet counts and
                        pairwise flavor agreement over a full ``Simulator`` run, and knows how to print itself and
                        serialize itself into the metadata saved alongside the histograms.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

from ..taggers.base import JetFlavorTagger
from ..taggers.flavor import JetFlavor

TaggerPair = Tuple[JetFlavorTagger, JetFlavorTagger]


@dataclass
class SimulationSummary:
    """
    Jet counts and pairwise flavor agreement accumulated over a full simulation run, saved as metadata alongside
    the histograms.
    """

    n_events: int
    n_jets: int
    tagger_counts: Dict[JetFlavorTagger, Dict[JetFlavor, int]]
    pairwise_agreement: Dict[TaggerPair, Dict[str, int]]

    def print_statistics(self) -> None:
        """
        Print the run's jet counts, any tagger-specific diagnostics, and, for every pair of taggers, how often
        they agree on flavor.
        """
        print(f"Number of events: {self.n_events}")
        print(f"Number of jets: {self.n_jets}")

        for tagger, counts in self.tagger_counts.items():
            diagnostics = tagger.get_diagnostics()
            diagnostics_str = "".join(f", {key}: {value}" for key, value in diagnostics.items())
            print(
                f"  [{tagger.identifier}] "
                f"quark: {counts[JetFlavor.QUARK]}, "
                f"gluon: {counts[JetFlavor.GLUON]}, "
                f"untagged: {counts[JetFlavor.UNTAGGED]}"
                f"{diagnostics_str}"
            )

        if self.pairwise_agreement:
            print("Pairwise flavor agreement (among jets both taggers tag):")

            for (tagger_a, tagger_b), counts in self.pairwise_agreement.items():
                both_tagged = counts["both_tagged"]
                fraction = counts["agree"] / both_tagged if both_tagged else float("nan")
                print(
                    f"  [{tagger_a.identifier} vs {tagger_b.identifier}] "
                    f"agree: {counts['agree']}/{both_tagged} ({100 * fraction:.1f}%)"
                )

    def to_dict(self) -> dict:
        """
        Build the run metadata entry: number of events, number of jets, and, for every tagger, its counts per
        flavor plus any tagger-specific diagnostics (see ``JetFlavorTagger.get_diagnostics``), plus every pair of
        taggers' flavor agreement.

        The pairwise agreement is a sanity check for tagger consistency: two taggers implementing genuinely
        different strategies should still agree on flavor most of the time for jets they both manage to tag, since
        the underlying hard-scattering ancestry is the same physical truth regardless of the algorithm used to
        recover it.

        Returns
        -------
        dict
            Metadata entry, saved under the ``"metadata"`` key of the output file.
        """
        return {
            "n_events": self.n_events,
            "n_jets": self.n_jets,
            "flavor_taggers": {
                tagger.identifier: {
                    "n_quark": counts[JetFlavor.QUARK],
                    "n_gluon": counts[JetFlavor.GLUON],
                    "n_untagged": counts[JetFlavor.UNTAGGED],
                    **tagger.get_diagnostics(),
                }
                for tagger, counts in self.tagger_counts.items()
            },
            "pairwise_flavor_agreement": [
                {
                    "taggers": [tagger_a.identifier, tagger_b.identifier],
                    "n_both_tagged": counts["both_tagged"],
                    "n_agree": counts["agree"],
                    "agreement_fraction": (
                        counts["agree"] / counts["both_tagged"] if counts["both_tagged"] else None
                    ),
                }
                for (tagger_a, tagger_b), counts in self.pairwise_agreement.items()
            ],
        }

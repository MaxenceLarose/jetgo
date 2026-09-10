"""
    @Title:             Writing your own observable.

    @Description:       Observables are the main extension point of jetgo. This example implements the jet
                        constituent multiplicity, a quantity that separates quark and gluon jets, and runs it
                        through the simulator exactly like a shipped observable.

                        The contract is six methods. ``fill`` receives one event's jets and adds them to a
                        histogram whose edges were fixed at construction; ``finalize`` is the classmethod that
                        turns the saved histogram into the normalized quantity, downstream.
"""

import json
from typing import Dict, List

import fastjet as fj
import numpy as np

from jetgo.observables.base import Observable
from jetgo.simulation import EventGenerator, JetFinder, ParticleSelector, Simulator
from jetgo.taggers import DescendancyTracing

BIN_EDGES = np.arange(0.5, 60.5, 2.0)
OUTPUT_PATH = "constituent_multiplicity.json"


class ConstituentMultiplicity(Observable):
    """
    Distribution of the number of constituents per jet, normalized to unit area.
    """

    N_JETS_KEY = "n_jets"
    BIN_EDGES_KEY = "bin_edges"

    def __init__(self, bin_edges: np.ndarray, pt_min: float = 1.0) -> None:
        self._bin_edges = np.asarray(bin_edges, dtype=float)
        self._counts = np.zeros(len(self._bin_edges) - 1, dtype=float)
        self._pt_min = pt_min
        self._n_jets = 0

    @property
    def identifier(self) -> str:
        # Shipped observables use an ObservableIdentifier member, whose value is the HEPData keyword. Any
        # hashable, unique value works: the simulator only uses it as this observable's key in the output.
        return "constituent_multiplicity"

    def fill(self, jets: List[fj.PseudoJet], cluster: fj.ClusterSequence) -> None:
        for jet in jets:
            self._n_jets += 1
            multiplicity = sum(1 for c in jet.constituents() if c.pt() > self._pt_min)

            index = int(np.searchsorted(self._bin_edges, multiplicity, side="right")) - 1
            if 0 <= index < len(self._counts):
                self._counts[index] += 1.0

    def scale(self, scale_factor: float) -> None:
        # Self-normalized, like the EEC: the generator's cross-section factor cancels in the shape.
        pass

    @classmethod
    def finalize(cls, values, bin_edges: np.ndarray, metadata: Dict) -> np.ndarray:
        counts = np.asarray(values, dtype=float)
        return counts / counts.sum() / np.diff(np.asarray(bin_edges, dtype=float))

    def _get_complementary_metadata(self) -> Dict:
        return {self.N_JETS_KEY: self._n_jets, self.BIN_EDGES_KEY: self._bin_edges.tolist()}

    def _get_binned_values(self) -> List[float]:
        return self._counts.tolist()


if __name__ == "__main__":
    simulator = Simulator(
        event_generator=EventGenerator(
            beam_1=2212, beam_2=2212, sqrt_s=5020, pt_min=80, pt_max=200, random_seed=42
        ),
        jet_finder=JetFinder(radius=0.4, pt_min=120, pt_max=140, max_abs_pseudorapidity=1.6),
        particle_selector=ParticleSelector(max_abs_pseudorapidity=2.4),
    )

    simulator.simulate(
        n_events=5_000,
        observables=ConstituentMultiplicity(bin_edges=BIN_EDGES),
        jet_flavor_taggers=DescendancyTracing(delta_r_max=0.4, use_pseudorapidity=True),
        output_path=OUTPUT_PATH,
        verbose=True,
    )

    histograms = json.loads(open(OUTPUT_PATH).read())["constituent_multiplicity"]["histograms"]
    centers = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

    for label, saved in (
        ("quark", histograms["descendancy_tracing"]["quark"]),
        ("gluon", histograms["descendancy_tracing"]["gluon"]),
    ):
        shape = ConstituentMultiplicity.finalize(saved["values"], BIN_EDGES, saved["metadata"])
        mean = float(np.sum(centers * shape * np.diff(BIN_EDGES)))
        print(f"{label:>6} jets: mean constituent multiplicity = {mean:.1f}")

    print("Gluon jets carry more constituents than quark jets: they radiate more, in the ratio of the")
    print("color charges C_A/C_F. The measured ratio is well below that naive 9/4, since hadronization")
    print("and the finite jet energy both dilute it.")

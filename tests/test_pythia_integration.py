"""
    @file:              test_pythia_integration.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       End-to-end tests that drive a real Pythia8 run through the full pipeline. They are
                        marked ``pythia`` and skipped where Pythia8 is unavailable, since its wheels are
                        Linux-only and it is an optional dependency.

                        The golden file these compare against was produced by this same configuration and is
                        committed alongside them, so a change that silently alters the physics fails here
                        rather than in someone's analysis months later.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jetgo._pythia", reason="Pythia8 is not installed")

from jetgo.observables import DoubleDifferentialJetCrossSection, EEC
from jetgo.simulation import EventGenerator, JetFinder, ParticleSelector, Simulator
from jetgo.taggers import DescendancyTracing
from jetgo.taggers.flavor import JetFlavor

pytestmark = pytest.mark.pythia

GOLDEN_FILE = Path(__file__).parent / "data" / "golden_run.json"

SEED = 20260909
N_EVENTS = 500
EEC_BIN_EDGES = np.geomspace(0.005, 0.8, 31)
CROSS_SECTION_BIN_EDGES = np.linspace(120.0, 140.0, 21)


def build_simulator() -> Simulator:
    """The exact configuration the golden file was produced with."""
    return Simulator(
        event_generator=EventGenerator(
            beam_1=2212, beam_2=2212, sqrt_s=5020, pt_min=80, pt_max=200, random_seed=SEED
        ),
        jet_finder=JetFinder(radius=0.4, pt_min=120, pt_max=140, max_abs_pseudorapidity=1.6),
        particle_selector=ParticleSelector(max_abs_pseudorapidity=2.4),
    )


def run_reference(output_path) -> dict:
    """Run the reference configuration and return its output."""
    build_simulator().simulate(
        n_events=N_EVENTS,
        observables=[
            EEC(pt_min=1, energy_weight=1, charged_only=True, use_pseudorapidity=True,
                bin_edges=EEC_BIN_EDGES),
            DoubleDifferentialJetCrossSection(max_abs_pseudorapidity=1.6,
                                              bin_edges=CROSS_SECTION_BIN_EDGES),
        ],
        output_path=output_path,
        jet_flavor_taggers=DescendancyTracing(delta_r_max=0.4, use_pseudorapidity=True),
    )
    return json.loads(Path(output_path).read_text())


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory) -> dict:
    return run_reference(tmp_path_factory.mktemp("run") / "run.json")


class TestReproducibility:

    def test_the_same_seed_gives_byte_identical_output(self, tmp_path):
        first = run_reference(tmp_path / "first.json")
        second = run_reference(tmp_path / "second.json")

        assert first == second

    def test_a_different_seed_gives_different_events(self, tmp_path):
        build_simulator()
        Simulator(
            event_generator=EventGenerator(
                beam_1=2212, beam_2=2212, sqrt_s=5020, pt_min=80, pt_max=200, random_seed=SEED + 1
            ),
            jet_finder=JetFinder(radius=0.4, pt_min=120, pt_max=140, max_abs_pseudorapidity=1.6),
            particle_selector=ParticleSelector(max_abs_pseudorapidity=2.4),
        ).simulate(
            n_events=N_EVENTS,
            observables=DoubleDifferentialJetCrossSection(
                max_abs_pseudorapidity=1.6, bin_edges=CROSS_SECTION_BIN_EDGES
            ),
            output_path=tmp_path / "other_seed.json",
        )
        other = json.loads((tmp_path / "other_seed.json").read_text())
        same = run_reference(tmp_path / "same_seed.json")

        assert other["metadata"]["n_jets"] != same["metadata"]["n_jets"]


class TestDescendancyTracing:

    def test_it_leaves_no_jet_untagged_when_the_fallback_is_enabled(self, reference_run):
        counts = reference_run["metadata"]["flavor_taggers"]["descendancy_tracing"]

        assert counts["n_untagged"] == 0
        assert counts["n_quark"] + counts["n_gluon"] == reference_run["metadata"]["n_jets"]

    def test_the_beam_remnant_fallback_stays_rare(self, reference_run):
        """
        Measured at 0.3% of jets over a 20,000-event sample, so 5% is a bound a real regression
        would cross while ordinary fluctuation in this small reference run would not.
        """
        metadata = reference_run["metadata"]
        fallback_calls = metadata["flavor_taggers"]["descendancy_tracing"]["fallback_call_count"]

        assert fallback_calls <= 0.05 * metadata["n_jets"]

    def test_it_can_leave_jets_untagged_when_the_fallback_is_disabled(self):
        """Without the fallback, beam-remnant jets have no hard parton to vote for them."""
        tagger = DescendancyTracing(delta_r_max=0.4, use_pseudorapidity=True, handle_untagged_jets=False)

        assert tagger.get_diagnostics()["fallback_call_count"] == 0

    def test_both_flavors_are_produced_in_a_hard_qcd_sample(self, reference_run):
        counts = reference_run["metadata"]["flavor_taggers"]["descendancy_tracing"]

        assert counts["n_quark"] > 0
        assert counts["n_gluon"] > 0


class TestFlavorConsistency:

    def test_flavor_histograms_sum_to_the_inclusive_one(self, reference_run):
        histograms = reference_run["D2SIG/DPT/DYRAP"]["histograms"]
        inclusive = np.asarray(histograms["inclusive"]["values"])
        quark = np.asarray(histograms["descendancy_tracing"]["quark"]["values"])
        gluon = np.asarray(histograms["descendancy_tracing"]["gluon"]["values"])

        # Every jet is tagged, so the two flavors partition the inclusive sample exactly.
        assert (quark + gluon).tolist() == pytest.approx(inclusive.tolist())

    def test_the_eec_finalizes_to_a_unit_normalized_shape(self, reference_run):
        saved = reference_run["Energy-energy correlator"]["histograms"]["inclusive"]

        shape = EEC.finalize(saved["values"], EEC_BIN_EDGES, saved["metadata"])
        integral = float(np.sum(shape * np.diff(EEC_BIN_EDGES)))

        # Pairs falling outside the ΔR grid still count towards W_pairs, so the integral is at most one.
        assert 0.0 < integral <= 1.0


class TestGoldenRun:

    def test_the_reference_configuration_still_reproduces_the_committed_run(self, reference_run):
        assert GOLDEN_FILE.exists(), (
            f"{GOLDEN_FILE} is missing. Regenerate it with tests/data/generate_golden_run.py."
        )

        golden = json.loads(GOLDEN_FILE.read_text())

        assert reference_run["metadata"] == golden["metadata"]
        for observable in (k for k in golden if k != "metadata"):
            assert reference_run[observable] == golden[observable], f"{observable} changed"

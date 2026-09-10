"""
    @Title:             Splitting a run by generator-level jet flavor.

    @Description:       The same run as ex01, with a flavor tagger attached. Every observable is filled three
                        times over: once with all jets, once with the jets the tagger calls quark-initiated,
                        and once with the ones it calls gluon-initiated. The second half of the script reads
                        the file back and turns the saved histograms into normalized observables.
"""

import json

import numpy as np

from jetgo.observables import DoubleDifferentialJetCrossSection, EEC
from jetgo.simulation import EventGenerator, JetFinder, ParticleSelector, Simulator
from jetgo.taggers import DescendancyTracing

OUTPUT_PATH = "flavor_tagged_run.json"
EEC_BIN_EDGES = np.geomspace(0.005, 0.8, 31)
CROSS_SECTION_BIN_EDGES = np.linspace(120.0, 140.0, 21)

if __name__ == "__main__":
    # ---------------------------------------------------------------------------------------------------- #
    #                                              The run                                                  #
    # ---------------------------------------------------------------------------------------------------- #
    simulator = Simulator(
        event_generator=EventGenerator(
            beam_1=2212, beam_2=2212, sqrt_s=5020, pt_min=80, pt_max=200, random_seed=42
        ),
        jet_finder=JetFinder(radius=0.4, pt_min=120, pt_max=140, max_abs_pseudorapidity=1.6),
        particle_selector=ParticleSelector(max_abs_pseudorapidity=2.4),
    )

    simulator.simulate(
        n_events=5_000,
        observables=[
            EEC(bin_edges=EEC_BIN_EDGES, pt_min=1, charged_only=True, use_pseudorapidity=True),
            DoubleDifferentialJetCrossSection(
                bin_edges=CROSS_SECTION_BIN_EDGES, max_abs_pseudorapidity=1.6
            ),
        ],
        # Pass a list here to run several taggers at once. The output metadata then also reports, for every
        # pair, how often they agreed on the jets both of them tagged.
        jet_flavor_taggers=DescendancyTracing(delta_r_max=0.4, use_pseudorapidity=True),
        output_path=OUTPUT_PATH,
        verbose=True,
    )

    # ---------------------------------------------------------------------------------------------------- #
    #                                          Reading it back                                              #
    # ---------------------------------------------------------------------------------------------------- #
    result = json.loads(open(OUTPUT_PATH).read())

    counts = result["metadata"]["flavor_taggers"]["descendancy_tracing"]
    print(f"quark-tagged jets: {counts['n_quark']}")
    print(f"gluon-tagged jets: {counts['n_gluon']}")
    print(f"untagged jets:     {counts['n_untagged']}")

    # finalize turns a saved histogram into the normalized observable. It is a classmethod, and it runs here,
    # never during the simulation itself.
    histograms = result["Energy-energy correlator"]["histograms"]
    for label, saved in (
        ("inclusive", histograms["inclusive"]),
        ("quark", histograms["descendancy_tracing"]["quark"]),
        ("gluon", histograms["descendancy_tracing"]["gluon"]),
    ):
        shape = EEC.finalize(saved["values"], EEC_BIN_EDGES, saved["metadata"])
        peak = EEC_BIN_EDGES[:-1][int(np.argmax(shape))]
        print(f"{label:>10} EEC peaks near ΔR = {peak:.3f}")

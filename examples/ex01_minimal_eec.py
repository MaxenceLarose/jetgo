"""
    @Title:             Minimal energy-energy correlator run.

    @Description:       The smallest useful jetgo run. Generate hard-QCD proton-proton events, keep the
                        charged final-state particles inside a tracker-like acceptance, cluster them into
                        anti-kT jets, accumulate the energy-energy correlator, and write everything to one
                        JSON file.
"""

import numpy as np

from jetgo.observables import EEC
from jetgo.simulation import EventGenerator, JetFinder, ParticleSelector, Simulator

if __name__ == "__main__":
    # ---------------------------------------------------------------------------------------------------- #
    #                                           The three pieces                                            #
    # ---------------------------------------------------------------------------------------------------- #
    event_generator = EventGenerator(
        beam_1=2212,        # proton
        beam_2=2212,        # proton
        sqrt_s=5020,        # GeV
        pt_min=80,          # partonic p̂T window, in GeV
        pt_max=200,
        random_seed=42,     # anything non-zero: 0 lets Pythia8 pick its own seed, and the run stops
    )                       # being reproducible

    jet_finder = JetFinder(
        radius=0.4,
        pt_min=120,
        pt_max=140,
        max_abs_pseudorapidity=1.6,
    )

    particle_selector = ParticleSelector(
        max_abs_pseudorapidity=2.4,
    )

    # ---------------------------------------------------------------------------------------------------- #
    #                                              The run                                                  #
    # ---------------------------------------------------------------------------------------------------- #
    simulator = Simulator(
        event_generator=event_generator,
        jet_finder=jet_finder,
        particle_selector=particle_selector,
    )

    # The bin edges are required. Observables histogram at fill time, so the file size does not grow with
    # the run length, and the edges have to be the ones the result will be read on.
    observable = EEC(
        bin_edges=np.geomspace(0.005, 0.8, 31),
        pt_min=1,
        energy_weight=1,
        charged_only=True,
        use_pseudorapidity=True,
    )

    simulator.simulate(
        n_events=5_000,
        observables=observable,
        output_path="jet_eec.json",
        verbose=True,
    )

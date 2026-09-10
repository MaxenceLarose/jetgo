# jetgo

*Jet Event Toolkit for Generating Observables.*

`jetgo` generates collider events, clusters the particles into jets, tags each jet with the color
charge of the parton that initiated it, and writes the observables out.

You define the simulation, the observables and the tagger. `jetgo` runs the loop.

# Notable features

- **One event loop, written once.** Generation, particle selection, jet finding, observable
  filling and cross-section normalization run in a single pass, and land in one JSON file
  alongside the run metadata needed to interpret it.
- **Generator-level flavor tagging.** Every jet can be labelled quark or gluon by tracing the
  hard partons' shower descendants forward, a definition that is immune to color reconnection.
  Taggers are pluggable, and running several at once gives you their pairwise agreement for free.
- **Observables that bin at fill time.** Cost is independent of run length, so a million-event
  window produces a file of kilobytes rather than gigabytes.
- **Built on the standard stack.** [PYTHIA8](https://pythia.org) for generation and
  [FastJet](https://fastjet.fr) via the Scikit-HEP bindings for clustering. `jetgo` adds the
  pipeline, not another wrapper around either.

# Installation

## Latest stable version

```bash
pip install jetgo[pythia]
```

## Latest (possibly unstable) version

```bash
pip install "jetgo[pythia] @ git+https://github.com/MaxenceLarose/jetgo"
```

PYTHIA8 is an optional dependency because the `pythia8mc` wheels are Linux-only. Plain
`pip install jetgo` works everywhere and gives you every observable and tagger; only event
generation needs the extra. If you compiled PYTHIA8 yourself with its Python interface enabled,
`jetgo` finds it and you do not need the extra at all.

# Quick usage preview

```python
import numpy as np

from jetgo.observables import EEC
from jetgo.simulation import EventGenerator, JetFinder, ParticleSelector, Simulator
from jetgo.taggers import DescendancyTracing

simulator = Simulator(
    event_generator=EventGenerator(
        beam_1=2212, beam_2=2212, sqrt_s=5020, pt_min=80, pt_max=200, random_seed=42
    ),
    jet_finder=JetFinder(radius=0.4, pt_min=120, pt_max=140, max_abs_pseudorapidity=1.6),
    particle_selector=ParticleSelector(max_abs_pseudorapidity=2.4),
)

simulator.simulate(
    n_events=100_000,
    observables=EEC(bin_edges=np.geomspace(0.005, 0.8, 31), charged_only=True, use_pseudorapidity=True),
    jet_flavor_taggers=DescendancyTracing(delta_r_max=0.4, use_pseudorapidity=True),
    output_path="jet_eec.json",
)
```

That run writes one file holding the inclusive energy-energy correlator, its quark-tagged and
gluon-tagged counterparts, and the run metadata: event and jet counts, per-flavor jet counts, and
the generator's cross-section normalization.

# Motivation

The pieces of a generator-level jet study are all available in Python. `pythia8mc` ships PYTHIA's
own bindings, Scikit-HEP's `fastjet` ships the clustering, and `awkward` and `uproot` handle what
comes after. What is missing is the part in between, and every group writes it again: loop over
events, cut on final-state particles, cluster, decide which jets to keep, fill a histogram,
remember to divide by `sigmaGen / weightSum`, and get the bookkeeping right when the same run has
to be split by jet flavor.

**Generator-level flavor tagging in particular has no reusable implementation.** Ghost
association lives inside the CMS and ATLAS C++ frameworks. The ATLAS FTAG Python stack starts
downstream, from jets whose truth labels already exist. There is nowhere to `pip install` a jet
flavor definition, compare it against another one on the same events, or hand a referee the exact
code that produced a label. `jetgo` is that place.

# How it works

## Main concepts

A run is three collaborating objects handed to a `Simulator`:

| object | responsibility |
|---|---|
| `EventGenerator` | configures PYTHIA8 for hard-QCD collisions in a p̂T window, and reports the cross-section normalization |
| `ParticleSelector` | keeps final-state, visible, optionally charged particles inside an acceptance, and converts them to FastJet inputs |
| `JetFinder` | holds a jet definition and the kinematic cuts a jet must pass |

You then hand `simulate` one or more `Observable` objects, and optionally one or more
`JetFlavorTagger` objects. Each observable is filled with every jet; each tagger additionally
produces a quark-tagged and a gluon-tagged copy of every observable, filled only with the jets it
labels accordingly.

`DescendancyTracing` never leaves a jet untagged. A tagger you write yourself might, and those
jets are excluded from both flavor histograms while still counting towards the inclusive one.

## Writing your own observable

Subclass `jetgo.observables.base.Observable` and implement `identifier`, `fill`, `scale`,
`finalize`, `_get_complementary_metadata` and `_get_binned_values`. Observables bin at fill time,
so `fill` receives one event's jets and adds their contributions to a histogram whose edges were
fixed at construction. `finalize` is the classmethod that turns the saved histogram into the
normalized quantity, and it runs downstream, never during the simulation.

## Writing your own tagger

Subclass `jetgo.taggers.base.JetFlavorTagger` and implement `identifier` and
`tag(jet, event) -> JetFlavor`. The event handed to `tag` is the untouched PYTHIA8 record, so the
full particle history is available. Pass several taggers to one run and the output metadata will
report, for every pair, how often they agree on the jets both of them tag.

## The shipped tagger

`DescendancyTracing` starts from the outgoing hard partons and walks their shower descendants
forward, letting the descendants that land near the jet axis vote with their transverse momentum.
Because it never reads a particle's mother list, color reconnection cannot mislead it. Jets whose
lineage traces back to the beam remnant rather than to the hard scattering are handled by an
explicit fallback, so no jet has to be discarded.

## Need more examples?

See the [`examples/`](examples) directory.

# License

This project is licensed under the terms of the [BSD 3-Clause License](LICENSE).

# Citation

If you use `jetgo` in your work, please cite the paper it was written for:

```bibtex
@article{Bossi:2026colors,
  author        = {Bossi, Hannah and Larose, Maxence and Mehtar-Tani, Yacine},
  title         = {The Colors of Jet Quenching},
  year          = {2026},
  eprint        = {2609.05609},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ph},
  url           = {https://arxiv.org/abs/2609.05609}
}
```

# Contact

Maxence Larose — [maxence.larose@stonybrook.edu](mailto:maxence.larose@stonybrook.edu)

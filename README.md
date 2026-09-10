<p align="center">
  <img src="https://raw.githubusercontent.com/MaxenceLarose/jetgo/main/images/jetgo_banner.png" alt="jetgo" width="820">
</p>

<p align="center">
  <i>The simplest way to generate collider events and extract jet observable data from them.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PYTHIA-8.3+-orange.svg" alt="PYTHIA 8.3+">
  <img src="https://img.shields.io/badge/FastJet-3.5+-9cf.svg" alt="FastJet 3.5+">
  <a href="https://github.com/MaxenceLarose/jetgo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-green.svg" alt="BSD-3-Clause license"></a>
  <a href="https://pypi.org/project/jetgo/"><img src="https://img.shields.io/pypi/dm/jetgo?label=downloads&color=blue" alt="Downloads"></a>
</p>

`jetgo` generates collider events, clusters the particles into jets, tags each jet with the color
charge of the parton that initiated it, and writes the observables out.

You define the simulator, the observables and the tagger. `jetgo` does the rest.

# Installation

```bash
pip install jetgo[pythia]
```

PYTHIA8 is an optional dependency, because its wheels are Linux-only. Plain `pip install jetgo`
works everywhere and gives you every observable and tagger; only event generation needs the extra.

# Quick usage preview

```python
import numpy as np

from jetgo.observables import EEC
from jetgo.simulation import EventGenerator, JetFinder, ParticleSelector, Simulator
from jetgo.taggers import DescendancyTracing

event_generator = EventGenerator(
    beam_1=2212,
    beam_2=2212,
    sqrt_s=5020,
    pt_min=80,
    pt_max=200,
    random_seed=42
)

particle_selector = ParticleSelector(
    max_abs_pseudorapidity=2.4
)

jet_finder = JetFinder(
    radius=0.4,
    pt_min=120,
    pt_max=140,
    max_abs_pseudorapidity=1.6
)

observable = EEC(
    bin_edges=np.geomspace(0.005, 0.8, 31),
    charged_only=True,
    use_pseudorapidity=True
)

jet_flavor_tagger = DescendancyTracing(
    delta_r_max=0.4,
    use_pseudorapidity=True
)

simulator = Simulator(
    event_generator=event_generator,
    particle_selector=particle_selector,
    jet_finder=jet_finder
)

simulator.simulate(
    n_events=100_000,
    observables=observable,
    jet_flavor_taggers=jet_flavor_tagger,
    output_path="jet_eec.json"
)
```

That run writes one file holding the inclusive energy-energy correlator, its quark-tagged and
gluon-tagged counterparts, and the run metadata.

# Documentation

Full documentation lives at [maxencelarose.github.io/jetgo](https://maxencelarose.github.io/jetgo),
including how to write your own observable or your own flavor tagger. Runnable scripts are in
[`examples/`](examples).

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

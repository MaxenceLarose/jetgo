# Examples

Each script is standalone and runnable. Install the package with the Pythia8 extra first:

```bash
pip install jetgo[pythia]
```

| script | what it shows |
|---|---|
| `ex01_minimal_eec.py` | the smallest useful run: generate events, cluster jets, fill one observable, save |
| `ex02_flavor_tagged_run.py` | the same run split into quark-tagged and gluon-tagged jets, and how to read the result back |
| `ex03_custom_observable.py` | writing your own `Observable` subclass and running it through the simulator |

They are deliberately small, a few thousand events each, so they finish in under a minute. Real
runs are the same code with a larger `n_events`.

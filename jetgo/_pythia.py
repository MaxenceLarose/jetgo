"""
    @file:              _pythia.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 09/2026

    @Description:       This file resolves and re-exports the Pythia8 module, accounting for the different module
                        name used by the pip-installed ``pythia8mc`` package versus a self-compiled ``pythia8``
                        build.

                        Pythia8 is an optional dependency of this package: only event generation and particle
                        selection need it at runtime, so observables and taggers import without it. Importing
                        this module on a machine without either build raises with an actionable message rather
                        than a bare ModuleNotFoundError from somewhere deep in the import graph.
"""

import importlib

# The module is named pythia8 when users compile it themselves, and pythia8mc when installed via pip.
_CANDIDATE_MODULE_NAMES = ("pythia8mc", "pythia8")

pythia8 = None
for _name in _CANDIDATE_MODULE_NAMES:
    try:
        pythia8 = importlib.import_module(_name)
        break
    except ModuleNotFoundError:
        continue

if pythia8 is None:
    raise ModuleNotFoundError(
        "Pythia8 is required for event generation but is not installed. Install it with "
        "`pip install jetgo[pythia]`, which pulls in the `pythia8mc` wheel, or compile Pythia8 yourself with its "
        "Python interface enabled so that `import pythia8` works. Note that the `pythia8mc` wheels are Linux-only; "
        "on another platform, a self-compiled build is the only option. The rest of jetgo, including every "
        "observable and tagger, imports and runs without Pythia8."
    )

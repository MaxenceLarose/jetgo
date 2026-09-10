"""
    @file:              conf.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       Sphinx configuration. The package's docstrings are NumPy-style throughout, so the API
                        reference is generated from them by autodoc and numpydoc rather than written by hand.
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(".."))

import jetgo

project = "jetgo"
author = "Maxence Larose"
copyright = f"{date.today().year}, {author}"
release = jetgo.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# Pythia8 is an optional dependency and is not installed in the docs build, which is deliberate: that the
# build succeeds is itself a check that observables and taggers import without it. Only the two simulation
# modules that genuinely need it at import time are mocked.
autodoc_mock_imports = ["pythia8mc", "pythia8"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_title = f"jetgo {release}"
html_theme_options = {
    "github_url": "https://github.com/MaxenceLarose/jetgo",
    "show_prev_next": False,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

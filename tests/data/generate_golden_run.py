"""
    @file:              generate_golden_run.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       Regenerates the golden run that ``test_pythia_integration.py`` compares against.

                        Run this only when the physics is meant to change, and say so in the commit message:
                        a regenerated golden file is the record that a change to the pipeline moved numbers.

                        Usage: python tests/data/generate_golden_run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_pythia_integration import GOLDEN_FILE, run_reference

if __name__ == "__main__":
    run_reference(GOLDEN_FILE)
    print(f"Wrote {GOLDEN_FILE}")

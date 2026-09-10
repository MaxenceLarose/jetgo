"""
    @file:              identifier.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 06/2026

    @Description:       This file defines the ObservableIdentifier enumeration. It provides a strongly-typed set of
                        identifiers used to reference HEPData observables in a consistent and error-safe way across the
                        codebase.
"""

from enum import StrEnum


class ObservableIdentifier(StrEnum):
    """
    Observable identifiers used by HEPData.
    """

    D2SIG_DPT_DYRAP = "D2SIG/DPT/DYRAP"
    EEC = "Energy-energy correlator"

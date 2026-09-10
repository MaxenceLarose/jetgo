"""
    @file:              identifier.py
    @Author:            Maxence Larose

    @Creation Date:     07/2026
    @Last modification: 07/2026

    @Description:       This file defines the JetFlavorTaggerIdentifier enumeration. It provides a strongly-typed set
                        of identifiers used to reference jet flavor tagging strategies in a consistent and error-safe
                        way across the codebase, e.g. as keys when saving results to disk.
"""

from enum import StrEnum


class JetFlavorTaggerIdentifier(StrEnum):
    """
    Identifiers for jet flavor tagging strategies.
    """

    DESCENDANCY_TRACING = "descendancy_tracing"

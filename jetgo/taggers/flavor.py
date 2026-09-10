"""
    @file:              flavor.py
    @Author:            Maxence Larose

    @Creation Date:     06/2026
    @Last modification: 07/2026

    @Description:       This file defines the JetFlavor enumeration, used to label a reconstructed jet as
                        originating from a quark, a gluon, or as untagged.
"""

from enum import StrEnum


class JetFlavor(StrEnum):
    GLUON = "gluon"
    QUARK = "quark"
    UNTAGGED = "untagged"

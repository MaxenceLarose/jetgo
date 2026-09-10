"""
    @file:              classes.py
    @Author:            Maxence Larose

    @Creation Date:     08/2026
    @Last modification: 08/2026

    @Description:       Maps each ``ObservableIdentifier`` to the ``Observable`` subclass that implements it.
"""

from .double_differential_jet_cross_section import DoubleDifferentialJetCrossSection
from .eec import EEC
from .identifier import ObservableIdentifier

OBSERVABLE_CLASSES = {
    ObservableIdentifier.D2SIG_DPT_DYRAP: DoubleDifferentialJetCrossSection,
    ObservableIdentifier.EEC: EEC,
}

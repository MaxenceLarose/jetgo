"""
    @file:              classes.py
    @Author:            Maxence Larose

    @Creation Date:     08/2026
    @Last modification: 09/2026

    @Description:       Maps each ``JetFlavorTaggerIdentifier`` to the ``JetFlavorTagger`` subclass that implements
                        it.
"""

from .descendancy_tracing import DescendancyTracing
from .identifier import JetFlavorTaggerIdentifier

JET_FLAVOR_TAGGER_CLASSES = {
    JetFlavorTaggerIdentifier.DESCENDANCY_TRACING: DescendancyTracing,
}

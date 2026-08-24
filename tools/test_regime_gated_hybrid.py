#!/usr/bin/env python3
"""Regression tests for the frozen prediction-prevalence regime gate."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.regime_gated_hybrid_eval import (
    PREVALENCE_GATE,
    choose_regime,
)


assert 0.00099123 < PREVALENCE_GATE < 0.00749227
assert choose_regime(0.00099123) == "conservative-veto"
assert choose_regime(0.00749227) == "recovery-add"
assert choose_regime(PREVALENCE_GATE) == "mapmos-abstain"
assert choose_regime(0.00322620) == "recovery-add"

print("Regime-gated hybrid regression tests passed")

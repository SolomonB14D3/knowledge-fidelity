"""Confidence cartography module for knowledge-fidelity.

Measures teacher-forced confidence: the probability a causal LM assigns to its
own tokens when fed text in a single forward pass. This reveals where the model
is uncertain, detecting false beliefs, Mandela effects, and contested claims.

Adapted from the Confidence Cartography project
(github.com/SolomonB14D3/confidence-cartography).

Key findings:
  - Confidence ratios correlate with human false-belief prevalence
    (rho=0.652, p=0.016, across Pythia 160M-12B)
  - Generalizes to out-of-domain medical claims (88% accuracy at 6.9B)
  - Mandela effect false memories show systematically lower confidence
"""

from .engine import analyze_confidence, load_model, unload_model
from .schema import TokenAnalysis, ConfidenceRecord, save_records, load_records

__all__ = [
    "analyze_confidence",
    "load_model",
    "unload_model",
    "TokenAnalysis",
    "ConfidenceRecord",
    "save_records",
    "load_records",
]

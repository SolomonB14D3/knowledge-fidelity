"""Data schema for confidence analysis records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class TokenAnalysis:
    """Per-token confidence metrics at a single position."""
    position: int               # 0-indexed position in the sequence
    token_id: int               # The actual token ID at this position
    token_str: str              # Decoded string for this token
    top1_prob: float            # Probability model assigned to this token
    top1_rank: int              # Rank in model's distribution (0 = top)
    entropy: float              # Entropy (bits) of full vocab distribution
    top5_tokens: list[str]      # Top 5 predicted token strings
    top5_probs: list[float]     # Their probabilities
    top5_ids: list[int]         # Their token IDs


@dataclass
class ConfidenceRecord:
    """Complete confidence analysis of one text."""
    text: str                   # The full input text
    category: str               # E.g. "true_fact", "false_belief", "mandela"
    label: str                  # Short ID, e.g. "capital_france"
    mode: str                   # "fixed" (teacher-forced) or "generated"
    num_tokens: int             # Total tokens in the text
    tokens: list[TokenAnalysis] # Per-token analysis (length = num_tokens - 1)
    mean_top1_prob: float       # Mean of top1_prob across all positions
    mean_entropy: float         # Mean entropy across all positions
    std_top1_prob: float        # Std dev of top1_prob
    std_entropy: float          # Std dev of entropy
    min_confidence_pos: int     # Position of lowest top1_prob
    min_confidence_token: str   # The token at that position
    min_confidence_value: float # The top1_prob at that position
    model_name: str = ""
    model_revision: str = "main"
    metadata: dict = field(default_factory=dict)


def save_records(records: list[ConfidenceRecord], path: Path) -> None:
    """Append records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def load_records(path: Path) -> list[ConfidenceRecord]:
    """Load records from a JSONL file."""
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["tokens"] = [TokenAnalysis(**t) for t in d["tokens"]]
            records.append(ConfidenceRecord(**d))
    return records

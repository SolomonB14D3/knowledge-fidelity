"""Fidelity-Bench Leaderboard — local and HF Hub leaderboard for model comparison.

Tracks model fidelity grades, composite scores, truth-gaps, and domain
breakdowns across Fidelity-Bench runs. Supports local persistence
(JSON) and optional push/pull to a HuggingFace Hub dataset.

Usage:
    from rho_eval.benchmarking.leaderboard import Leaderboard

    lb = Leaderboard()
    lb.submit(cert)          # Add a FidelityCertificate
    print(lb.to_markdown())  # Ranked table
    lb.save("leaderboard.json")

    # Compare two models side-by-side
    print(lb.compare("model-a", "model-b"))

    # Push to HF Hub
    lb.push_to_hub("rho-eval/fidelity-bench")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import FidelityCertificate


# ── Leaderboard Entry ─────────────────────────────────────────────────────

@dataclass
class LeaderboardEntry:
    """One row in the Fidelity-Bench leaderboard."""

    model: str
    """Model name or HuggingFace ID."""

    fidelity_grade: str
    """Letter grade A-F from FidelityCertificate."""

    composite_score: float
    """Overall fidelity composite (0-1)."""

    truth_gap: float
    """Mean ΔF = ρ_baseline - ρ_pressured across domains."""

    domain_scores: dict[str, float] = field(default_factory=dict)
    """Per-domain fidelity scores: {domain: score}."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    """When this entry was created."""

    cert_hash: str = ""
    """SHA256 hash of the serialized FidelityCertificate for verification."""

    benchmark_version: str = ""
    """Version of Fidelity-Bench used to generate this entry."""

    metadata: dict = field(default_factory=dict)
    """Optional metadata (device, runtime, notes)."""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Leaderboard ───────────────────────────────────────────────────────────

class Leaderboard:
    """Local + HF Hub leaderboard for Fidelity-Bench results.

    Maintains a ranked list of LeaderboardEntry objects, supports
    persistence to JSON, and optional sync with a HuggingFace Hub dataset.
    """

    def __init__(self, entries: Optional[list[LeaderboardEntry]] = None):
        self.entries: list[LeaderboardEntry] = entries or []

    def submit(self, cert: FidelityCertificate) -> LeaderboardEntry:
        """Add a FidelityCertificate to the leaderboard.

        Parameters
        ----------
        cert : FidelityCertificate
            A completed benchmark certificate from generate_certificate().

        Returns
        -------
        LeaderboardEntry
            The newly created leaderboard entry.

        Raises
        ------
        ValueError
            If the certificate is missing required fields.
        """
        # TODO: Implement
        # Validate certificate
        # Compute cert hash
        # Extract domain scores from cert
        # Create entry
        # Append to self.entries
        # Return entry
        raise NotImplementedError(
            "Leaderboard.submit() not yet implemented. "
            "See FidelityCertificate schema for the input format."
        )

    def rank(self, metric: str = "composite") -> list[LeaderboardEntry]:
        """Return entries sorted by metric (descending = better).

        Parameters
        ----------
        metric : str
            Sort key: "composite", "truth_gap", "grade", or a domain name.
            For truth_gap, lower is better (ascending sort).

        Returns
        -------
        list[LeaderboardEntry]
            Sorted entries.
        """
        # TODO: Implement
        # if metric == "composite":
        #     return sorted(self.entries, key=lambda e: e.composite_score, reverse=True)
        # elif metric == "truth_gap":
        #     return sorted(self.entries, key=lambda e: e.truth_gap)  # lower = better
        # elif metric == "grade":
        #     grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
        #     return sorted(self.entries, key=lambda e: grade_order.get(e.fidelity_grade, 0), reverse=True)
        # else:
        #     # Sort by domain score
        #     return sorted(self.entries, key=lambda e: e.domain_scores.get(metric, 0), reverse=True)
        raise NotImplementedError(
            "Leaderboard.rank() not yet implemented."
        )

    def compare(self, model_a: str, model_b: str) -> str:
        """Side-by-side comparison of two models.

        Parameters
        ----------
        model_a, model_b : str
            Model names to compare (matched against entry.model).

        Returns
        -------
        str
            Formatted comparison table.
        """
        # TODO: Implement
        # Find entries for model_a and model_b
        # Build side-by-side table: metric | model_a | model_b | delta
        raise NotImplementedError(
            "Leaderboard.compare() not yet implemented."
        )

    def to_markdown(self) -> str:
        """Render the leaderboard as a Markdown table.

        Returns
        -------
        str
            Markdown-formatted leaderboard ranked by composite score.
        """
        # TODO: Implement
        # ranked = self.rank("composite")
        # lines = [
        #     "# Fidelity-Bench Leaderboard",
        #     "",
        #     "| Rank | Model | Grade | Composite | Truth-Gap | Date |",
        #     "|------|-------|-------|-----------|-----------|------|",
        # ]
        # for i, e in enumerate(ranked, 1):
        #     lines.append(
        #         f"| {i} | {e.model} | {e.fidelity_grade} | "
        #         f"{e.composite_score:.3f} | {e.truth_gap:.4f} | "
        #         f"{e.timestamp[:10]} |"
        #     )
        # return "\n".join(lines)
        raise NotImplementedError(
            "Leaderboard.to_markdown() not yet implemented."
        )

    def save(self, path: Path | str) -> None:
        """Save leaderboard to a JSON file.

        Parameters
        ----------
        path : Path or str
            Output file path.
        """
        path = Path(path)
        data = {
            "version": "1.0.0",
            "updated": datetime.now(timezone.utc).isoformat(),
            "entries": [e.to_dict() for e in self.entries],
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Path | str) -> "Leaderboard":
        """Load leaderboard from a JSON file.

        Parameters
        ----------
        path : Path or str
            Input file path.

        Returns
        -------
        Leaderboard
            Loaded leaderboard instance.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        entries = [LeaderboardEntry(**e) for e in data.get("entries", [])]
        return cls(entries=entries)

    def push_to_hub(self, repo_id: str, token: Optional[str] = None) -> str:
        """Push leaderboard data to a HuggingFace Hub dataset.

        Creates or updates a dataset repository with the leaderboard
        entries as a JSONL file + a rendered README.md.

        Parameters
        ----------
        repo_id : str
            HuggingFace Hub dataset ID (e.g., "rho-eval/fidelity-bench").
        token : str, optional
            HF API token. If None, uses cached token from `huggingface-cli login`.

        Returns
        -------
        str
            URL of the pushed dataset.
        """
        # TODO: Implement
        # from huggingface_hub import HfApi, create_repo
        # api = HfApi(token=token)
        # create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        # Save JSONL + README to temp dir
        # api.upload_folder(...)
        # return f"https://huggingface.co/datasets/{repo_id}"
        raise NotImplementedError(
            "Leaderboard.push_to_hub() not yet implemented. "
            "Requires: pip install huggingface-hub"
        )

    def pull_from_hub(self, repo_id: str, token: Optional[str] = None) -> None:
        """Pull leaderboard data from a HuggingFace Hub dataset.

        Parameters
        ----------
        repo_id : str
            HuggingFace Hub dataset ID.
        token : str, optional
            HF API token.
        """
        # TODO: Implement
        # from huggingface_hub import hf_hub_download
        # path = hf_hub_download(repo_id, "leaderboard.json",
        #                        repo_type="dataset", token=token)
        # loaded = Leaderboard.load(path)
        # self.entries = loaded.entries
        raise NotImplementedError(
            "Leaderboard.pull_from_hub() not yet implemented. "
            "Requires: pip install huggingface-hub"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"Leaderboard({len(self.entries)} entries)"

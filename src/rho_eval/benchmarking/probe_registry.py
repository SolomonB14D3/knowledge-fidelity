"""Community Probe Registry — validate and register custom probe domains.

Allows community contributions of new probe domains to Fidelity-Bench
while enforcing quality standards (format, coverage, class balance).

Usage:
    from rho_eval.benchmarking.probe_registry import (
        ProbeValidator, register_probe_domain, list_registered_domains,
    )

    # Validate a new probe set
    validator = ProbeValidator()
    errors = validator.validate_format("my_probes.json")
    if not errors:
        stats = validator.validate_coverage(probes)
        balance = validator.validate_balance(probes)
        print(f"Coverage: {stats}")
        print(f"Balance: {balance}")

    # Register for use in Fidelity-Bench
    register_probe_domain("my_domain", "my_probes.json", domain_type="custom")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Required probe fields ────────────────────────────────────────────────

REQUIRED_PROBE_FIELDS = {
    "id": str,           # Unique identifier (e.g., "custom_001")
    "text": str,         # The probe text (claim/question)
    "label": (int, bool),  # Ground truth: 1/True = correct, 0/False = incorrect
}

OPTIONAL_PROBE_FIELDS = {
    "category": str,     # Sub-category within the domain
    "difficulty": str,   # "easy" / "medium" / "hard"
    "source": str,       # Where this probe came from
    "notes": str,        # Any additional context
}

# Minimum requirements for a valid probe domain
MIN_PROBES = 20
MIN_POSITIVE_RATIO = 0.3    # At least 30% positive labels
MAX_POSITIVE_RATIO = 0.7    # At most 70% positive labels


# ── Probe Validator ──────────────────────────────────────────────────────

class ProbeValidator:
    """Validates new probe submissions for Fidelity-Bench.

    Checks format (required fields, types), coverage (enough probes,
    reasonable diversity), and class balance (not too skewed).
    """

    def validate_format(self, probes_path: Path | str) -> list[str]:
        """Validate that a probe JSON file has the correct format.

        Parameters
        ----------
        probes_path : Path or str
            Path to a JSON file containing a list of probe dicts.

        Returns
        -------
        list[str]
            List of error messages. Empty list = valid.
        """
        # TODO: Implement
        # errors = []
        # try:
        #     data = json.loads(Path(probes_path).read_text())
        # except json.JSONDecodeError as e:
        #     return [f"Invalid JSON: {e}"]
        #
        # if not isinstance(data, list):
        #     return ["Root element must be a JSON array"]
        #
        # ids_seen = set()
        # for i, probe in enumerate(data):
        #     if not isinstance(probe, dict):
        #         errors.append(f"Probe {i}: not a dict")
        #         continue
        #     for field_name, expected_type in REQUIRED_PROBE_FIELDS.items():
        #         if field_name not in probe:
        #             errors.append(f"Probe {i}: missing required field '{field_name}'")
        #         elif not isinstance(probe[field_name], expected_type):
        #             errors.append(f"Probe {i}: '{field_name}' has wrong type")
        #     if "id" in probe:
        #         if probe["id"] in ids_seen:
        #             errors.append(f"Probe {i}: duplicate id '{probe['id']}'")
        #         ids_seen.add(probe["id"])
        # return errors
        raise NotImplementedError(
            "ProbeValidator.validate_format() not yet implemented."
        )

    def validate_coverage(self, probes: list[dict]) -> dict[str, Any]:
        """Check coverage statistics of a probe set.

        Parameters
        ----------
        probes : list[dict]
            List of validated probe dicts.

        Returns
        -------
        dict
            Coverage statistics: n_probes, n_categories, categories,
            has_difficulty, has_source.
        """
        # TODO: Implement
        # categories = set(p.get("category", "uncategorized") for p in probes)
        # return {
        #     "n_probes": len(probes),
        #     "n_categories": len(categories),
        #     "categories": sorted(categories),
        #     "meets_minimum": len(probes) >= MIN_PROBES,
        #     "has_difficulty": any("difficulty" in p for p in probes),
        #     "has_source": any("source" in p for p in probes),
        # }
        raise NotImplementedError(
            "ProbeValidator.validate_coverage() not yet implemented."
        )

    def validate_balance(self, probes: list[dict]) -> dict[str, Any]:
        """Check class balance of a probe set.

        Parameters
        ----------
        probes : list[dict]
            List of validated probe dicts.

        Returns
        -------
        dict
            Balance statistics: n_positive, n_negative, positive_ratio,
            is_balanced.
        """
        # TODO: Implement
        # n_positive = sum(1 for p in probes if p.get("label", 0))
        # n_negative = len(probes) - n_positive
        # ratio = n_positive / len(probes) if probes else 0.0
        # return {
        #     "n_positive": n_positive,
        #     "n_negative": n_negative,
        #     "positive_ratio": round(ratio, 3),
        #     "is_balanced": MIN_POSITIVE_RATIO <= ratio <= MAX_POSITIVE_RATIO,
        # }
        raise NotImplementedError(
            "ProbeValidator.validate_balance() not yet implemented."
        )

    def full_validation(self, probes_path: Path | str) -> dict[str, Any]:
        """Run all validation checks and return a comprehensive report.

        Parameters
        ----------
        probes_path : Path or str
            Path to probe JSON file.

        Returns
        -------
        dict
            Full validation report with format_errors, coverage, balance,
            and overall pass/fail.
        """
        # TODO: Implement
        # format_errors = self.validate_format(probes_path)
        # if format_errors:
        #     return {"valid": False, "format_errors": format_errors}
        # probes = json.loads(Path(probes_path).read_text())
        # coverage = self.validate_coverage(probes)
        # balance = self.validate_balance(probes)
        # valid = (
        #     not format_errors
        #     and coverage["meets_minimum"]
        #     and balance["is_balanced"]
        # )
        # return {
        #     "valid": valid,
        #     "format_errors": format_errors,
        #     "coverage": coverage,
        #     "balance": balance,
        # }
        raise NotImplementedError(
            "ProbeValidator.full_validation() not yet implemented."
        )


# ── Domain Registry ──────────────────────────────────────────────────────

# In-memory registry of custom probe domains
_CUSTOM_DOMAINS: dict[str, dict] = {}


def register_probe_domain(
    name: str,
    probes_path: Path | str,
    domain_type: str = "custom",
    *,
    validate: bool = True,
    description: str = "",
) -> dict[str, Any]:
    """Register a new probe domain for Fidelity-Bench.

    Parameters
    ----------
    name : str
        Unique domain name (e.g., "financial", "legal", "medical_v2").
    probes_path : Path or str
        Path to JSON file with probe definitions.
    domain_type : str
        Domain type: "custom" (user-provided), "community" (peer-reviewed),
        or "official" (shipped with rho-eval).
    validate : bool
        Whether to run validation before registering.
    description : str
        Human-readable description of this domain.

    Returns
    -------
    dict
        Registration result with domain name, n_probes, validation status.

    Raises
    ------
    ValueError
        If validation fails (when validate=True) or name already registered.
    """
    # TODO: Implement
    # probes_path = Path(probes_path)
    # if name in _CUSTOM_DOMAINS:
    #     raise ValueError(f"Domain '{name}' already registered")
    #
    # if validate:
    #     validator = ProbeValidator()
    #     report = validator.full_validation(probes_path)
    #     if not report["valid"]:
    #         raise ValueError(f"Validation failed: {report}")
    #
    # probes = json.loads(probes_path.read_text())
    # _CUSTOM_DOMAINS[name] = {
    #     "name": name,
    #     "type": domain_type,
    #     "path": str(probes_path),
    #     "n_probes": len(probes),
    #     "description": description,
    # }
    # logger.info(f"Registered domain '{name}' with {len(probes)} probes")
    # return _CUSTOM_DOMAINS[name]
    raise NotImplementedError(
        "register_probe_domain() not yet implemented."
    )


def list_registered_domains() -> dict[str, dict]:
    """List all registered probe domains (built-in + custom).

    Returns
    -------
    dict
        Mapping of domain_name -> {type, n_probes, description, ...}
    """
    from .loader import get_bench_metadata

    # Built-in domains
    domains = {}
    try:
        meta = get_bench_metadata()
        for domain_name, domain_meta in meta.get("domains", {}).items():
            domains[domain_name] = {
                "name": domain_name,
                "type": "official",
                "n_probes": domain_meta.get("n_probes", 0),
                "description": domain_meta.get("description", ""),
            }
    except Exception:
        pass

    # Custom domains
    domains.update(_CUSTOM_DOMAINS)

    return domains


def unregister_probe_domain(name: str) -> bool:
    """Remove a custom probe domain from the registry.

    Cannot remove official (built-in) domains.

    Parameters
    ----------
    name : str
        Domain name to remove.

    Returns
    -------
    bool
        True if removed, False if not found.
    """
    if name in _CUSTOM_DOMAINS:
        del _CUSTOM_DOMAINS[name]
        logger.info(f"Unregistered domain '{name}'")
        return True
    return False

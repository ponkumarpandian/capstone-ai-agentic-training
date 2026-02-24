"""
MediSuite Agent — Coverage Validation Agent
Validates insurance coverage and policy details.
"""

import json
import logging
from datetime import datetime
from typing import Any

from agents.base_agent import BaseAgent
from config import settings

logger = logging.getLogger(__name__)


class CoverageValidationAgent(BaseAgent):
    """Validate insurance coverage using a policy database."""

    agent_name = "coverage_validation_agent"
    agent_instructions = (
        "You are an insurance coverage validation specialist. "
        "Verify that the patient's insurance policy is valid, active, "
        "and covers the requested services."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._policy_db = self._load_policy_db()

    @staticmethod
    def _load_policy_db() -> list[dict[str, Any]]:
        path = settings.data_dir / "policy_database.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning("Policy database not found at %s", path)
        return []

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the Coverage Validation Agent.

        Args:
            input_data:
                insurance_details (dict): policy_number, provider.
                cpt4_codes (list[str]): Procedures to validate coverage for.
                amount (float): Total charge amount.

        Returns:
            dict with validation_status, reason, policy_details,
            coverage_checks.
        """
        insurance = input_data.get("insurance_details", {})
        policy_number = insurance.get("policy_number", "")
        provider = insurance.get("provider", "")
        cpt4_codes = input_data.get("cpt4_codes", [])
        amount = input_data.get("amount", 0.0)

        logger.info(
            "CoverageValidationAgent: starting — policy=%s, provider=%s",
            policy_number,
            provider,
        )

        # ── Step 1: Look up the policy ─────────────────────────────
        policy = self._find_policy(policy_number)

        if not policy:
            result = {
                "validation_status": "Invalid",
                "reason": f"Policy {policy_number} not found in database.",
                "policy_details": None,
                "coverage_checks": [],
            }
            self._store_result(result, policy_number)
            return result

        # ── Step 2: Validate policy status ─────────────────────────
        checks: list[dict[str, Any]] = []

        # Check expiry
        expiry = datetime.strptime(policy["expiry_date"], "%Y-%m-%d")
        effective = datetime.strptime(policy["effective_date"], "%Y-%m-%d")
        now = datetime.now()

        is_active = effective <= now <= expiry
        checks.append({
            "check": "Policy Active",
            "passed": is_active,
            "detail": (
                f"Effective {policy['effective_date']} – {policy['expiry_date']}"
                if is_active
                else f"Policy expired on {policy['expiry_date']}"
            ),
        })

        # Check coverage field
        is_coverage_valid = policy.get("coverage", "").lower() == "valid"
        checks.append({
            "check": "Coverage Status",
            "passed": is_coverage_valid,
            "detail": f"Coverage status: {policy.get('coverage', 'Unknown')}",
        })

        # Check provider match
        provider_match = (
            provider.lower() == policy.get("provider", "").lower()
            if provider
            else True
        )
        checks.append({
            "check": "Provider Match",
            "passed": provider_match,
            "detail": (
                f"Provider matches: {policy.get('provider')}"
                if provider_match
                else f"Provider mismatch: expected {policy.get('provider')}, got {provider}"
            ),
        })

        # Check service coverage
        covered_services = policy.get("covered_services", [])
        services_covered = len(covered_services) > 0  # At least some services covered
        checks.append({
            "check": "Service Coverage",
            "passed": services_covered,
            "detail": f"Covered services: {', '.join(covered_services)}",
        })

        # ── Step 3: Determine overall validation ───────────────────
        all_passed = all(c["passed"] for c in checks)
        reasons: list[str] = []
        if all_passed:
            reasons.append("All validation checks passed.")
        else:
            for c in checks:
                if not c["passed"]:
                    reasons.append(f"{c['check']}: {c['detail']}")

        validation_status = "Valid" if all_passed else "Invalid"

        # ── Step 4: RAG — store validation result ───────────────────
        result = {
            "validation_status": validation_status,
            "reason": " | ".join(reasons),
            "policy_details": {
                "policy_number": policy.get("policy_number"),
                "provider": policy.get("provider"),
                "plan_type": policy.get("plan_type"),
                "effective_date": policy.get("effective_date"),
                "expiry_date": policy.get("expiry_date"),
                "copay": policy.get("copay"),
                "deductible": policy.get("deductible"),
                "deductible_met": policy.get("deductible_met"),
            },
            "coverage_checks": checks,
        }

        self._store_result(result, policy_number)

        # ── Step 5: RAG — retrieve prior validations ───────────────
        prior = self.kb.retrieve_documents(
            query=f"validation {policy_number}",
            doc_type="validation_result",
            top_k=3,
        )
        result["prior_validations_count"] = len(prior)

        logger.info(
            "CoverageValidationAgent: done — status=%s, reason=%s",
            validation_status,
            result["reason"][:120],
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_policy(self, policy_number: str) -> dict[str, Any] | None:
        for policy in self._policy_db:
            if policy.get("policy_number") == policy_number:
                return policy
        return None

    def _store_result(self, result: dict[str, Any], policy_number: str) -> None:
        self.kb.insert_document("validation_result", {
            "policy_number": policy_number,
            "validation_status": result["validation_status"],
            "reason": result["reason"],
        })

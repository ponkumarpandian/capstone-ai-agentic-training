"""
MediSuite Agent — Triage Agent
Makes approve / deny / review decisions based on validation and extracted data.
"""

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TriageAgent(BaseAgent):
    """Decide whether to approve, deny, or flag a claim for review."""

    agent_name = "triage_agent"
    agent_instructions = (
        "You are a medical claims triage specialist. Based on the validation "
        "results, extracted data, and any prior similar claims, decide whether "
        "this claim should be APPROVED, DENIED, or sent for REVIEW. "
        "Return a JSON object with keys: "
        '"decision" (Approve | Deny | Review) and '
        '"justification" (string explaining the reasoning).'
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the Triage Agent.

        Args:
            input_data:
                validation_status (str): Valid / Invalid.
                validation_reason (str): Reason from coverage validation.
                coverage_checks (list[dict]): Individual check results.
                icd10_codes (list[str]): Diagnosis codes.
                cpt4_codes (list[str]): Procedure codes.
                amount (float): Total charge.
                claim_id (str): The generated claim ID.
                patient_id (str): Patient identifier.

        Returns:
            dict with decision, justification, risk_factors, confidence.
        """
        validation_status = input_data.get("validation_status", "Unknown")
        validation_reason = input_data.get("validation_reason", "")
        coverage_checks = input_data.get("coverage_checks", [])
        icd10_codes = input_data.get("icd10_codes", [])
        cpt4_codes = input_data.get("cpt4_codes", [])
        amount = input_data.get("amount", 0.0)
        claim_id = input_data.get("claim_id", "")
        patient_id = input_data.get("patient_id", "")

        logger.info(
            "TriageAgent: starting — claim_id=%s, validation=%s",
            claim_id,
            validation_status,
        )

        # ── Step 1: RAG — retrieve similar past claims ─────────────
        prior_claims = self.kb.retrieve_documents(
            query=f"triage decision ICD {' '.join(icd10_codes)}",
            doc_type="triage_decision",
            top_k=5,
        )

        # ── Step 2: Try AI-based triage ────────────────────────────
        ai_decision = self._ai_triage(input_data, prior_claims)

        if ai_decision.get("decision"):
            decision = ai_decision["decision"]
            justification = ai_decision.get("justification", "AI-based decision.")
        else:
            # Fallback to rule-based triage
            decision, justification = self._rule_based_triage(
                validation_status,
                validation_reason,
                coverage_checks,
                icd10_codes,
                cpt4_codes,
                amount,
            )

        # ── Step 3: Calculate risk factors ─────────────────────────
        risk_factors = self._assess_risk(
            validation_status, coverage_checks, amount, icd10_codes, cpt4_codes
        )
        confidence = self._calculate_confidence(risk_factors)

        # ── Step 4: RAG — insert triage decision ───────────────────
        triage_result = {
            "claim_id": claim_id,
            "patient_id": patient_id,
            "decision": decision,
            "justification": justification,
            "risk_factors": risk_factors,
            "confidence": confidence,
            "icd10_codes": icd10_codes,
            "cpt4_codes": cpt4_codes,
            "amount": amount,
        }
        self.kb.insert_document("triage_decision", triage_result)

        logger.info(
            "TriageAgent: done — decision=%s, confidence=%.1f%%",
            decision,
            confidence * 100,
        )
        return triage_result

    # ------------------------------------------------------------------
    # AI-based triage
    # ------------------------------------------------------------------
    def _ai_triage(
        self,
        input_data: dict[str, Any],
        prior_claims: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._client is None:
            return {}

        prior_context = ""
        if prior_claims:
            prior_context = (
                "\n\nSimilar past triage decisions:\n"
                + json.dumps(prior_claims[:3], indent=2, default=str)
            )

        prompt = (
            "Make a triage decision for the following medical claim:\n\n"
            f"Validation Status: {input_data.get('validation_status')}\n"
            f"Validation Reason: {input_data.get('validation_reason')}\n"
            f"ICD-10 Codes: {json.dumps(input_data.get('icd10_codes', []))}\n"
            f"CPT-4 Codes: {json.dumps(input_data.get('cpt4_codes', []))}\n"
            f"Amount: ${input_data.get('amount', 0):.2f}\n"
            f"{prior_context}\n\n"
            "Return JSON with: decision (Approve|Deny|Review), justification (string)."
        )
        response = self._ask_ai(prompt)
        if response:
            return self._parse_json_response(response)
        return {}

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------
    def _rule_based_triage(
        self,
        validation_status: str,
        validation_reason: str,
        coverage_checks: list[dict[str, Any]],
        icd10_codes: list[str],
        cpt4_codes: list[str],
        amount: float,
    ) -> tuple[str, str]:
        """Apply deterministic rules to decide claim outcome."""
        reasons: list[str] = []

        # Rule 1: Invalid coverage → Deny
        if validation_status.lower() != "valid":
            return "Deny", f"Coverage validation failed: {validation_reason}"

        # Rule 2: No diagnosis codes → Review
        if not icd10_codes:
            reasons.append("No ICD-10 diagnosis codes provided.")

        # Rule 3: No procedure codes → Review
        if not cpt4_codes:
            reasons.append("No CPT-4 procedure codes provided.")

        # Rule 4: High-value claims → Review
        if amount > 1000:
            reasons.append(f"High-value claim (${amount:,.2f}).")

        # Rule 5: Check for failed coverage sub-checks
        failed_checks = [c for c in coverage_checks if not c.get("passed", True)]
        if failed_checks:
            for fc in failed_checks:
                reasons.append(f"Failed check: {fc.get('check')} – {fc.get('detail')}")

        if reasons:
            return "Review", " | ".join(reasons)

        return "Approve", "All validations passed. Coverage is active, codes are present, and amount is within normal range."

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------
    def _assess_risk(
        self,
        validation_status: str,
        coverage_checks: list[dict[str, Any]],
        amount: float,
        icd10_codes: list[str],
        cpt4_codes: list[str],
    ) -> list[dict[str, Any]]:
        """Identify risk factors for the claim."""
        factors: list[dict[str, Any]] = []

        if validation_status.lower() != "valid":
            factors.append({
                "factor": "Invalid Coverage",
                "severity": "high",
                "detail": "Insurance coverage validation failed.",
            })

        if amount > 500:
            severity = "high" if amount > 1000 else "medium"
            factors.append({
                "factor": "High Charge Amount",
                "severity": severity,
                "detail": f"Charge amount ${amount:,.2f} exceeds normal range.",
            })

        if not icd10_codes:
            factors.append({
                "factor": "Missing Diagnosis Codes",
                "severity": "high",
                "detail": "No ICD-10 codes present on claim.",
            })

        if not cpt4_codes:
            factors.append({
                "factor": "Missing Procedure Codes",
                "severity": "high",
                "detail": "No CPT-4 codes present on claim.",
            })

        if len(icd10_codes) > 4:
            factors.append({
                "factor": "Multiple Diagnoses",
                "severity": "low",
                "detail": f"{len(icd10_codes)} diagnosis codes — may need review.",
            })

        return factors

    def _calculate_confidence(self, risk_factors: list[dict[str, Any]]) -> float:
        """Calculate a confidence score between 0 and 1."""
        if not risk_factors:
            return 0.95

        severity_weights = {"high": 0.3, "medium": 0.15, "low": 0.05}
        penalty = sum(
            severity_weights.get(f.get("severity", "low"), 0.05)
            for f in risk_factors
        )
        return max(0.1, 1.0 - penalty)

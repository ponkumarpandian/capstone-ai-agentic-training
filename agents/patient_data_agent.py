"""
MediSuite Agent — Patient Data Agent
Validates patient information and extracts diagnoses/procedures from clinical notes.
"""

import json
import logging
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PatientDataAgent(BaseAgent):
    """Collect, validate, and enrich patient information."""

    agent_name = "patient_data_agent"
    agent_instructions = (
        "You are a medical data extraction specialist. "
        "Given clinical notes, extract diagnoses and procedures as structured JSON. "
        "Return ONLY a JSON object with keys: "
        '"diagnoses" (list of diagnosis strings) and '
        '"procedures" (list of procedure strings). '
        "If none are found, return empty lists."
    )

    # Required fields for validation
    REQUIRED_FIELDS = ["patient_id", "name", "dob", "insurance_details"]
    REQUIRED_INSURANCE_FIELDS = ["policy_number", "provider"]

    def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the Patient Data Agent.

        Args:
            input_data:
                patient_info (dict): Patient information JSON.
                clinical_notes (str): Raw clinical notes text.

        Returns:
            dict with validated_patient, diagnoses, procedures, validation_errors.
        """
        patient_info = input_data.get("patient_info", {})
        clinical_notes = input_data.get("clinical_notes", "")

        logger.info("PatientDataAgent: starting — patient_id=%s", patient_info.get("patient_id"))

        # ── Step 1: Validate patient info ──────────────────────────
        validation_errors = self._validate_patient(patient_info)
        is_valid = len(validation_errors) == 0

        # ── Step 2: Extract diagnoses & procedures from notes ──────
        extraction = self._extract_from_notes(clinical_notes)
        diagnoses = extraction.get("diagnoses", [])
        procedures = extraction.get("procedures", [])

        # ── Step 3: RAG — store validated data ──────────────────────
        self.kb.insert_document("patient_data", {
            "patient_id": patient_info.get("patient_id"),
            "name": patient_info.get("name"),
            "dob": patient_info.get("dob"),
            "insurance_provider": patient_info.get("insurance_details", {}).get("provider"),
            "is_valid": is_valid,
        })

        # ── Step 4: RAG — retrieve any prior records ───────────────
        prior_records = self.kb.retrieve_documents(
            query=f"patient {patient_info.get('name', '')}",
            doc_type="patient_data",
            top_k=3,
        )

        result = {
            "validated_patient": patient_info,
            "is_valid": is_valid,
            "validation_errors": validation_errors,
            "diagnoses": diagnoses,
            "procedures": procedures,
            "prior_records_count": len(prior_records),
        }

        logger.info(
            "PatientDataAgent: done — valid=%s, diagnoses=%s, procedures=%s",
            is_valid,
            diagnoses,
            procedures,
        )
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_patient(self, patient: dict[str, Any]) -> list[str]:
        """Validate required fields and return a list of error messages."""
        errors: list[str] = []

        for field in self.REQUIRED_FIELDS:
            if not patient.get(field):
                errors.append(f"Missing required field: {field}")

        insurance = patient.get("insurance_details", {})
        if insurance:
            for field in self.REQUIRED_INSURANCE_FIELDS:
                if not insurance.get(field):
                    errors.append(f"Missing required insurance field: {field}")
        return errors

    # ------------------------------------------------------------------
    # Diagnosis / procedure extraction
    # ------------------------------------------------------------------
    def _extract_from_notes(self, notes: str) -> dict[str, Any]:
        """Use Azure AI to extract diagnoses & procedures from clinical notes.

        Falls back to keyword-based extraction when Azure is unavailable.
        """
        if not notes.strip():
            return {"diagnoses": [], "procedures": []}

        # Try AI extraction first
        if self._client is not None:
            prompt = (
                "Analyse the following clinical notes and extract all diagnoses "
                "and medical procedures mentioned. Return the result as JSON with "
                'keys "diagnoses" (list[str]) and "procedures" (list[str]).\n\n'
                f"Clinical Notes:\n{notes}"
            )
            response = self._ask_ai(prompt)
            if response:
                parsed = self._parse_json_response(response)
                if parsed:
                    return parsed

        # ── Local fallback: keyword-based extraction ───────────────
        logger.info("PatientDataAgent: using local extraction fallback")
        return self._local_extract(notes)

    def _local_extract(self, notes: str) -> dict[str, Any]:
        """Simple keyword-based extraction as a fallback."""
        notes_lower = notes.lower()
        diagnoses: list[str] = []
        procedures: list[str] = []

        # Common diagnosis keywords
        diagnosis_keywords = {
            "influenza": "Influenza",
            "flu": "Influenza",
            "bronchitis": "Acute bronchitis",
            "pneumonia": "Pneumonia",
            "covid": "COVID-19",
            "hypertension": "Hypertension",
            "diabetes": "Diabetes mellitus",
            "fever": "Fever",
            "cough": "Cough",
            "upper respiratory infection": "Upper respiratory infection",
        }

        # Common procedure keywords
        procedure_keywords = {
            "rapid influenza test": "Rapid influenza diagnostic test",
            "rapid test": "Rapid diagnostic test",
            "chest x-ray": "Chest X-ray",
            "x-ray": "X-ray",
            "blood test": "Blood test",
            "venipuncture": "Venipuncture",
            "vaccination": "Vaccination",
            "immunization": "Immunization",
        }

        for keyword, diagnosis in diagnosis_keywords.items():
            if keyword in notes_lower and diagnosis not in diagnoses:
                diagnoses.append(diagnosis)

        for keyword, procedure in procedure_keywords.items():
            if keyword in notes_lower and procedure not in procedures:
                procedures.append(procedure)

        # Check for explicit "diagnosed with" pattern
        if "diagnosed with" in notes_lower:
            idx = notes_lower.index("diagnosed with") + len("diagnosed with")
            snippet = notes[idx:idx + 100].strip().split(".")[0].strip()
            if snippet and snippet not in diagnoses:
                diagnoses.append(snippet)

        return {"diagnoses": diagnoses, "procedures": procedures}

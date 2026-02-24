"""
MediSuite Agent — Document Code Agent
Extracts structured data from documents and looks up ICD-10 / CPT-4 codes.
"""

import json
import logging
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from config import settings

logger = logging.getLogger(__name__)


class DocumentCodeAgent(BaseAgent):
    """Extract structured data from documents and perform code lookups."""

    agent_name = "document_code_agent"
    agent_instructions = (
        "You are a medical coding specialist. Given clinical information "
        "including diagnoses and procedures, determine the correct ICD-10 and "
        "CPT-4 codes. Return a JSON object with keys: "
        '"icd10_codes" (list of code strings), '
        '"cpt4_codes" (list of code strings), '
        '"amount" (total charge as a number).'
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._icd10_db = self._load_code_db("icd10_codes.json")
        self._cpt4_db = self._load_code_db("cpt4_codes.json")

    # ------------------------------------------------------------------
    # Code database loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_code_db(filename: str) -> dict[str, Any]:
        path = settings.data_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning("Code database '%s' not found at %s", filename, path)
        return {}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the Document Code Agent.

        Args:
            input_data:
                patient_info (dict): Validated patient data.
                diagnoses (list[str]): Extracted diagnoses.
                procedures (list[str]): Extracted procedures.
                document_metadata (dict): Metadata about the source document.
                clinical_notes (str): Raw clinical notes (optional).

        Returns:
            dict with icd10_codes, cpt4_codes, amount, extracted_data.
        """
        diagnoses = input_data.get("diagnoses", [])
        procedures = input_data.get("procedures", [])
        patient_info = input_data.get("patient_info", {})
        document_metadata = input_data.get("document_metadata", {})
        clinical_notes = input_data.get("clinical_notes", "")

        logger.info("DocumentCodeAgent: starting — %d diagnoses, %d procedures",
                     len(diagnoses), len(procedures))

        # ── Step 1: Try AI-based code lookup ───────────────────────
        ai_codes = self._ai_code_lookup(diagnoses, procedures, clinical_notes)

        # ── Step 2: Local code lookup fallback / enrichment ────────
        icd10_codes = ai_codes.get("icd10_codes", [])
        cpt4_codes = ai_codes.get("cpt4_codes", [])
        amount = ai_codes.get("amount", 0.0)

        if not icd10_codes:
            icd10_codes = self._local_icd10_lookup(diagnoses)
        if not cpt4_codes:
            cpt4_codes = self._local_cpt4_lookup(procedures)
        if not amount:
            amount = self._calculate_amount(cpt4_codes)

        # ── Step 3: Build extracted data ───────────────────────────
        extracted_data = {
            "patient_id": patient_info.get("patient_id", ""),
            "provider": patient_info.get("insurance_details", {}).get(
                "provider", ""
            ),
            "icd10_codes": icd10_codes,
            "cpt4_codes": cpt4_codes,
            "amount": amount,
            "document_id": document_metadata.get("document_id", ""),
        }

        # ── Step 4: RAG — insert codes and extracted data ──────────
        self.kb.insert_document("document_metadata", {
            **document_metadata,
            "icd10_codes": icd10_codes,
            "cpt4_codes": cpt4_codes,
        })

        for code in icd10_codes:
            code_info = self._icd10_db.get(code, {"code": code, "description": ""})
            self.kb.insert_document("icd10_code", {
                "code_type": "ICD-10",
                **code_info,
            })

        for code in cpt4_codes:
            code_info = self._cpt4_db.get(code, {"code": code, "description": ""})
            self.kb.insert_document("cpt4_code", {
                "code_type": "CPT-4",
                **code_info,
            })

        # ── Step 5: RAG — retrieve prior claims for context ────────
        prior_claims = self.kb.retrieve_documents(
            query=f"patient {patient_info.get('patient_id', '')} claim",
            doc_type="claim",
            top_k=3,
        )

        result = {
            "icd10_codes": icd10_codes,
            "cpt4_codes": cpt4_codes,
            "amount": amount,
            "extracted_data": extracted_data,
            "icd10_details": [
                self._icd10_db.get(c, {"code": c}) for c in icd10_codes
            ],
            "cpt4_details": [
                self._cpt4_db.get(c, {"code": c}) for c in cpt4_codes
            ],
            "prior_claims_count": len(prior_claims),
        }

        logger.info(
            "DocumentCodeAgent: done — ICD-10=%s, CPT-4=%s, amount=$%.2f",
            icd10_codes,
            cpt4_codes,
            amount,
        )
        return result

    # ------------------------------------------------------------------
    # AI-based code lookup
    # ------------------------------------------------------------------
    def _ai_code_lookup(
        self,
        diagnoses: list[str],
        procedures: list[str],
        clinical_notes: str,
    ) -> dict[str, Any]:
        if self._client is None:
            return {}

        available_icd10 = json.dumps(
            {k: v.get("description", "") for k, v in self._icd10_db.items()}
        )
        available_cpt4 = json.dumps(
            {k: v.get("description", "") for k, v in self._cpt4_db.items()}
        )

        prompt = (
            "Given the following medical information, determine the correct "
            "ICD-10 and CPT-4 codes.\n\n"
            f"Diagnoses: {json.dumps(diagnoses)}\n"
            f"Procedures: {json.dumps(procedures)}\n"
            f"Clinical Notes:\n{clinical_notes[:1000]}\n\n"
            f"Available ICD-10 codes: {available_icd10}\n\n"
            f"Available CPT-4 codes: {available_cpt4}\n\n"
            "Return JSON with: icd10_codes (list), cpt4_codes (list), amount (number)."
        )
        response = self._ask_ai(prompt)
        if response:
            return self._parse_json_response(response)
        return {}

    # ------------------------------------------------------------------
    # Local fallback lookups
    # ------------------------------------------------------------------
    def _local_icd10_lookup(self, diagnoses: list[str]) -> list[str]:
        """Match diagnoses to ICD-10 codes via keyword search."""
        codes: list[str] = []
        for diagnosis in diagnoses:
            diag_lower = diagnosis.lower()
            for code, info in self._icd10_db.items():
                desc = info.get("description", "").lower()
                if any(word in desc for word in diag_lower.split() if len(word) > 3):
                    if code not in codes:
                        codes.append(code)
        # Default to J10.1 for influenza mentions
        if not codes and any("influenza" in d.lower() or "flu" in d.lower() for d in diagnoses):
            codes.append("J10.1")
        return codes

    def _local_cpt4_lookup(self, procedures: list[str]) -> list[str]:
        """Match procedures to CPT-4 codes via keyword search."""
        codes: list[str] = []
        for procedure in procedures:
            proc_lower = procedure.lower()
            for code, info in self._cpt4_db.items():
                desc = info.get("description", "").lower()
                if any(word in desc for word in proc_lower.split() if len(word) > 3):
                    if code not in codes:
                        codes.append(code)
        # Default: office visit if no specific match
        if not codes:
            codes.append("99213")
        # If rapid test mentioned, add influenza test code
        if any("rapid" in p.lower() or "influenza" in p.lower() for p in procedures):
            if "87804" not in codes:
                codes.append("87804")
        return codes

    def _calculate_amount(self, cpt4_codes: list[str]) -> float:
        """Sum base rates from CPT-4 code database."""
        total = 0.0
        for code in cpt4_codes:
            info = self._cpt4_db.get(code, {})
            total += info.get("base_rate", 0.0)
        return total if total > 0 else 200.00  # Default amount

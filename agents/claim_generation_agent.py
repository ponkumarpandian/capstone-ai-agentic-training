"""
MediSuite Agent — Claim Generation Agent
Generates CMS-1500 claim forms as PDFs and uploads them to Azure Blob Storage.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from agents.base_agent import BaseAgent
from config import settings
from storage.blob_storage import BlobStorageClient
from utils.pdf_generator import generate_cms1500_pdf

logger = logging.getLogger(__name__)


class ClaimGenerationAgent(BaseAgent):
    """Generate CMS-1500 claim forms and upload to blob storage."""

    agent_name = "claim_generation_agent"
    agent_instructions = (
        "You are a medical claims specialist responsible for generating "
        "CMS-1500 claim forms. Ensure all required fields are populated "
        "and the form is compliant with CMS standards."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._blob_client = BlobStorageClient()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the Claim Generation Agent.

        Args:
            input_data:
                patient_info (dict): Validated patient data.
                diagnoses (list[str]): Extracted diagnoses.
                procedures (list[str]): Extracted procedures.
                icd10_codes (list[str]): ICD-10 codes.
                cpt4_codes (list[str]): CPT-4 codes.
                amount (float): Total charge.
                validation_status (str): Coverage validation result.
                clinical_notes (str): Original clinical notes (optional, for provider info).

        Returns:
            dict with claim_id, pdf_path, blob_url, claim_metadata.
        """
        patient_info = input_data.get("patient_info", {})
        diagnoses = input_data.get("diagnoses", [])
        procedures = input_data.get("procedures", [])
        icd10_codes = input_data.get("icd10_codes", [])
        cpt4_codes = input_data.get("cpt4_codes", [])
        amount = input_data.get("amount", 0.0)
        clinical_notes = input_data.get("clinical_notes", "")

        claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"

        logger.info(
            "ClaimGenerationAgent: starting — claim_id=%s, patient=%s",
            claim_id,
            patient_info.get("patient_id"),
        )

        # ── Step 1: RAG — retrieve prior claim templates ───────────
        prior_templates = self.kb.retrieve_documents(
            query="CMS-1500 claim template",
            doc_type="claim",
            top_k=2,
        )

        # ── Step 2: Build claim data ───────────────────────────────
        provider_info = self._extract_provider_info(clinical_notes)

        claim_data = {
            "claim_id": claim_id,
            "patient_id": patient_info.get("patient_id", ""),
            "name": patient_info.get("name", ""),
            "dob": patient_info.get("dob", ""),
            "gender": patient_info.get("gender", ""),
            "address": patient_info.get("address", {}),
            "phone": patient_info.get("phone", ""),
            "insurance": patient_info.get("insurance_details", {}),
            "provider": provider_info.get("provider", patient_info.get("insurance_details", {}).get("provider", "")),
            "provider_npi": provider_info.get("npi", ""),
            "facility": provider_info.get("facility", ""),
            "date_of_service": provider_info.get("date_of_service", datetime.now().strftime("%Y-%m-%d")),
            "diagnoses": diagnoses,
            "procedures": procedures,
            "icd10_codes": icd10_codes,
            "cpt4_codes": cpt4_codes,
            "amount": amount,
        }

        # ── Step 3: Generate PDF ───────────────────────────────────
        pdf_filename = f"{claim_id}.pdf"
        pdf_path = settings.output_dir / pdf_filename
        generate_cms1500_pdf(claim_data, pdf_path)

        # ── Step 4: Upload to Blob Storage ─────────────────────────
        blob_url = self._blob_client.upload_file(
            file_path=pdf_path,
            blob_name=f"claims/{pdf_filename}",
        )

        # ── Step 5: RAG — insert claim metadata ───────────────────
        claim_metadata = {
            "claim_id": claim_id,
            "patient_id": patient_info.get("patient_id", ""),
            "provider": claim_data["provider"],
            "icd10_codes": icd10_codes,
            "cpt4_codes": cpt4_codes,
            "amount": amount,
            "pdf_path": str(pdf_path),
            "blob_url": blob_url,
            "generated_at": datetime.now().isoformat(),
        }
        self.kb.insert_document("claim", claim_metadata)

        result = {
            "claim_id": claim_id,
            "pdf_path": str(pdf_path),
            "blob_url": blob_url,
            "claim_metadata": claim_metadata,
        }

        logger.info(
            "ClaimGenerationAgent: done — claim_id=%s, pdf=%s",
            claim_id,
            pdf_path,
        )
        return result

    # ------------------------------------------------------------------
    # Provider info extraction
    # ------------------------------------------------------------------
    def _extract_provider_info(self, clinical_notes: str) -> dict[str, str]:
        """Extract provider, NPI, facility from clinical notes text."""
        info: dict[str, str] = {}
        if not clinical_notes:
            return info

        for line in clinical_notes.splitlines():
            line_stripped = line.strip()
            if line_stripped.lower().startswith("provider:"):
                info["provider"] = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.lower().startswith("npi:"):
                info["npi"] = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.lower().startswith("facility:"):
                info["facility"] = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.lower().startswith("date of visit:"):
                info["date_of_service"] = line_stripped.split(":", 1)[1].strip()

        return info

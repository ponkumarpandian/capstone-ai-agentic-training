"""
MediSuite Agent — Workflow Orchestrator
Coordinates all five agents in a sequential pipeline.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from agents.patient_data_agent import PatientDataAgent
from agents.document_code_agent import DocumentCodeAgent
from agents.coverage_validation_agent import CoverageValidationAgent
from agents.claim_generation_agent import ClaimGenerationAgent
from agents.triage_agent import TriageAgent
from rag.knowledge_base import RAGKnowledgeBase

logger = logging.getLogger(__name__)


class MediSuiteOrchestrator:
    """Orchestrate the MediSuite multi-agent claim processing workflow.

    Pipeline:
        1. Patient Data Agent → validate patient, extract diagnoses
        2. Document Code Agent → look up ICD-10 / CPT-4 codes
        3. Coverage Validation Agent → verify insurance coverage
        4. Claim Generation Agent → generate CMS-1500 PDF
        5. Triage Agent → approve / deny / review the claim
    """

    def __init__(self) -> None:
        self.kb = RAGKnowledgeBase()
        self.patient_agent = PatientDataAgent(knowledge_base=self.kb)
        self.document_agent = DocumentCodeAgent(knowledge_base=self.kb)
        self.coverage_agent = CoverageValidationAgent(knowledge_base=self.kb)
        self.claim_agent = ClaimGenerationAgent(knowledge_base=self.kb)
        self.triage_agent = TriageAgent(knowledge_base=self.kb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_workflow(
        self,
        patient_data_path: str | Path,
        clinical_notes_path: str | Path,
        document_metadata_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute the full MediSuite claim processing workflow.

        Args:
            patient_data_path: Path to JSON file with patient information.
            clinical_notes_path: Path to text file with clinical notes.
            document_metadata_path: Optional path to document metadata JSON.

        Returns:
            Complete workflow result with each agent's output.
        """
        workflow_start = time.time()
        results: dict[str, Any] = {"steps": {}, "errors": []}

        # ── Load input files ───────────────────────────────────────
        try:
            patient_info = self._load_json(patient_data_path)
            clinical_notes = Path(clinical_notes_path).read_text(encoding="utf-8")
            document_metadata = (
                self._load_json(document_metadata_path)
                if document_metadata_path
                else {"document_id": "auto", "uploaded_by": "system"}
            )
        except Exception as exc:
            results["errors"].append(f"Failed to load input files: {exc}")
            results["status"] = "error"
            return results

        # ── Step 1: Patient Data Agent ─────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1/5 — Patient Data Agent")
        logger.info("=" * 60)
        step_start = time.time()
        try:
            patient_result = self.patient_agent.run({
                "patient_info": patient_info,
                "clinical_notes": clinical_notes,
            })
            patient_result["duration_s"] = round(time.time() - step_start, 2)
            results["steps"]["patient_data"] = patient_result
        except Exception as exc:
            logger.error("Patient Data Agent failed: %s", exc)
            results["errors"].append(f"Patient Data Agent: {exc}")
            results["steps"]["patient_data"] = {"error": str(exc)}
            patient_result = {
                "diagnoses": [],
                "procedures": [],
                "is_valid": False,
            }

        # ── Step 2: Document Code Agent ────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 2/5 — Document Code Agent")
        logger.info("=" * 60)
        step_start = time.time()
        try:
            document_result = self.document_agent.run({
                "patient_info": patient_info,
                "diagnoses": patient_result.get("diagnoses", []),
                "procedures": patient_result.get("procedures", []),
                "document_metadata": document_metadata,
                "clinical_notes": clinical_notes,
            })
            document_result["duration_s"] = round(time.time() - step_start, 2)
            results["steps"]["document_code"] = document_result
        except Exception as exc:
            logger.error("Document Code Agent failed: %s", exc)
            results["errors"].append(f"Document Code Agent: {exc}")
            results["steps"]["document_code"] = {"error": str(exc)}
            document_result = {
                "icd10_codes": [],
                "cpt4_codes": [],
                "amount": 0.0,
            }

        # ── Step 3: Coverage Validation Agent ──────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3/5 — Coverage Validation Agent")
        logger.info("=" * 60)
        step_start = time.time()
        try:
            coverage_result = self.coverage_agent.run({
                "insurance_details": patient_info.get("insurance_details", {}),
                "cpt4_codes": document_result.get("cpt4_codes", []),
                "amount": document_result.get("amount", 0.0),
            })
            coverage_result["duration_s"] = round(time.time() - step_start, 2)
            results["steps"]["coverage_validation"] = coverage_result
        except Exception as exc:
            logger.error("Coverage Validation Agent failed: %s", exc)
            results["errors"].append(f"Coverage Validation Agent: {exc}")
            results["steps"]["coverage_validation"] = {"error": str(exc)}
            coverage_result = {"validation_status": "Unknown", "reason": str(exc)}

        # ── Step 4: Claim Generation Agent ─────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 4/5 — Claim Generation Agent")
        logger.info("=" * 60)
        step_start = time.time()
        try:
            claim_result = self.claim_agent.run({
                "patient_info": patient_info,
                "diagnoses": patient_result.get("diagnoses", []),
                "procedures": patient_result.get("procedures", []),
                "icd10_codes": document_result.get("icd10_codes", []),
                "cpt4_codes": document_result.get("cpt4_codes", []),
                "amount": document_result.get("amount", 0.0),
                "validation_status": coverage_result.get("validation_status", ""),
                "clinical_notes": clinical_notes,
            })
            claim_result["duration_s"] = round(time.time() - step_start, 2)
            results["steps"]["claim_generation"] = claim_result
        except Exception as exc:
            logger.error("Claim Generation Agent failed: %s", exc)
            results["errors"].append(f"Claim Generation Agent: {exc}")
            results["steps"]["claim_generation"] = {"error": str(exc)}
            claim_result = {"claim_id": "ERROR"}

        # ── Step 5: Triage Agent ───────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 5/5 — Triage Agent")
        logger.info("=" * 60)
        step_start = time.time()
        try:
            triage_result = self.triage_agent.run({
                "validation_status": coverage_result.get("validation_status", ""),
                "validation_reason": coverage_result.get("reason", ""),
                "coverage_checks": coverage_result.get("coverage_checks", []),
                "icd10_codes": document_result.get("icd10_codes", []),
                "cpt4_codes": document_result.get("cpt4_codes", []),
                "amount": document_result.get("amount", 0.0),
                "claim_id": claim_result.get("claim_id", ""),
                "patient_id": patient_info.get("patient_id", ""),
            })
            triage_result["duration_s"] = round(time.time() - step_start, 2)
            results["steps"]["triage"] = triage_result
        except Exception as exc:
            logger.error("Triage Agent failed: %s", exc)
            results["errors"].append(f"Triage Agent: {exc}")
            results["steps"]["triage"] = {"error": str(exc)}

        # ── Summary ────────────────────────────────────────────────
        total_duration = round(time.time() - workflow_start, 2)
        results["status"] = "completed" if not results["errors"] else "completed_with_errors"
        results["total_duration_s"] = total_duration
        results["summary"] = {
            "patient": patient_info.get("name", "Unknown"),
            "claim_id": claim_result.get("claim_id", "N/A"),
            "decision": results.get("steps", {}).get("triage", {}).get("decision", "N/A"),
            "justification": results.get("steps", {}).get("triage", {}).get("justification", "N/A"),
            "amount": document_result.get("amount", 0.0),
            "pdf_path": claim_result.get("pdf_path", "N/A"),
        }

        logger.info("=" * 60)
        logger.info("WORKFLOW COMPLETE — %s in %.2fs", results["status"], total_duration)
        logger.info("=" * 60)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(path: str | Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

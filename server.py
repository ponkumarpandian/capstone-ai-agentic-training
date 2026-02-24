"""
MediSuite Agent — FastAPI Web Server
Provides a web interface and REST API for the claim processing workflow.
"""

import json
import logging
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from orchestrator import MediSuiteOrchestrator
from agents.chat_handler import ChatHandler

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("azure").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── FastAPI App ────────────────────────────────────────────────────
app = FastAPI(
    title="MediSuite Agent",
    description="Multi-Agent Medical Claim Processing System",
    version="1.0.0",
)

orchestrator = MediSuiteOrchestrator()
chat_handler = ChatHandler(knowledge_base=orchestrator.kb)

# ══════════════════════════════════════════════════════════════════
# In-memory stores (production would use a database)
# ══════════════════════════════════════════════════════════════════
_chat_histories: dict[str, list[dict[str, str]]] = {}
_workflow_results: dict[str, Any] = {}
_audit_log: list[dict[str, Any]] = []
_knowledge_entries: list[dict[str, Any]] = []
_settings_store: dict[str, Any] = {
    "project_endpoint": settings.project_endpoint,
    "model_deployment_name": settings.model_deployment_name,
    "blob_container_name": settings.blob_container_name,
    "search_index_name": settings.search_index_name,
    "azure_ai_configured": settings.validate_azure_ai(),
    "blob_storage_configured": settings.validate_blob_storage(),
    "search_configured": settings.validate_search(),
}


def _add_audit(action: str, claim_id: str = "", details: str = "", user: str = "system") -> None:
    """Append an entry to the in-memory audit log."""
    _audit_log.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "claim_id": claim_id,
        "details": details,
        "user": user,
    })


def _add_knowledge(doc_type: str, data: dict[str, Any], source: str = "") -> None:
    """Append an entry to the in-memory knowledge base."""
    _knowledge_entries.append({
        "id": str(uuid.uuid4()),
        "doc_type": doc_type,
        "data": data,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ══════════════════════════════════════════════════════════════════
# HTML + SPA  Route
# ══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the SPA web interface."""
    template_path = settings.project_root / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(500, "Web interface template not found.")
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════
# Workflow API
# ══════════════════════════════════════════════════════════════════

@app.post("/api/workflow")
async def run_workflow(
    patient_file: UploadFile = File(...),
    notes_file: UploadFile = File(...),
    metadata_file: UploadFile | None = File(None),
):
    """Upload files and trigger the claim processing workflow."""
    tmpdir = Path(tempfile.mkdtemp(prefix="medisuite_"))
    try:
        patient_path = tmpdir / "patient.json"
        notes_path = tmpdir / "notes.txt"
        patient_path.write_bytes(await patient_file.read())
        notes_path.write_bytes(await notes_file.read())

        metadata_path = None
        if metadata_file is not None:
            metadata_path = tmpdir / "metadata.json"
            metadata_path.write_bytes(await metadata_file.read())

        _add_audit("workflow_started", details="Workflow initiated via web UI")

        results = orchestrator.run_workflow(
            patient_data_path=str(patient_path),
            clinical_notes_path=str(notes_path),
            document_metadata_path=str(metadata_path) if metadata_path else None,
        )

        claim_id = results.get("summary", {}).get("claim_id", "unknown")
        results["created_at"] = datetime.now(timezone.utc).isoformat()
        _workflow_results[claim_id] = results

        # Audit
        decision = results.get("steps", {}).get("triage", {}).get("decision", "N/A")
        _add_audit("workflow_completed", claim_id=claim_id,
                   details=f"Decision: {decision}")

        # Populate knowledge base entries from agent outputs
        _populate_knowledge(results, claim_id)

        return results
    except Exception as exc:
        logger.exception("Workflow failed")
        _add_audit("workflow_failed", details=str(exc))
        raise HTTPException(500, detail=str(exc))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _populate_knowledge(results: dict[str, Any], claim_id: str) -> None:
    """Extract knowledge entries from workflow results."""
    steps = results.get("steps", {})

    # Patient data
    patient = steps.get("patient_data", {})
    if patient and "error" not in patient:
        _add_knowledge("patient_data", {
            "patient": patient.get("validated_patient", {}).get("name", ""),
            "diagnoses": patient.get("diagnoses", []),
            "procedures": patient.get("procedures", []),
        }, source=claim_id)

    # Codes
    doc = steps.get("document_code", {})
    if doc and "error" not in doc:
        for detail in doc.get("icd10_details", []):
            _add_knowledge("icd10_code", detail, source=claim_id)
        for detail in doc.get("cpt4_details", []):
            _add_knowledge("cpt4_code", detail, source=claim_id)

    # Validation
    cov = steps.get("coverage_validation", {})
    if cov and "error" not in cov:
        _add_knowledge("validation_result", {
            "status": cov.get("validation_status"),
            "reason": cov.get("reason"),
            "policy": cov.get("policy_details", {}),
        }, source=claim_id)

    # Triage
    tri = steps.get("triage", {})
    if tri and "error" not in tri:
        _add_knowledge("triage_decision", {
            "decision": tri.get("decision"),
            "justification": tri.get("justification"),
            "confidence": tri.get("confidence"),
        }, source=claim_id)


# ══════════════════════════════════════════════════════════════════
# Claims API
# ══════════════════════════════════════════════════════════════════

@app.get("/api/claims")
async def list_claims():
    """List all processed claims."""
    claims = []
    for cid, result in _workflow_results.items():
        summary = result.get("summary", {})
        steps = result.get("steps", {})
        claims.append({
            "claim_id": cid,
            "patient": summary.get("patient", "Unknown"),
            "decision": steps.get("triage", {}).get("decision", "N/A"),
            "amount": summary.get("amount", 0),
            "status": result.get("status", "unknown"),
            "created_at": result.get("created_at", ""),
            "duration_s": result.get("total_duration_s", 0),
            "icd10_codes": steps.get("document_code", {}).get("icd10_codes", []),
            "cpt4_codes": steps.get("document_code", {}).get("cpt4_codes", []),
            "validation_status": steps.get("coverage_validation", {}).get("validation_status", ""),
            "pdf_path": steps.get("claim_generation", {}).get("pdf_path", ""),
        })
    return {"claims": claims, "total": len(claims)}


@app.get("/api/claims/{claim_id}")
async def get_claim(claim_id: str):
    """Retrieve a single claim with full details."""
    if claim_id in _workflow_results:
        return _workflow_results[claim_id]
    raise HTTPException(404, detail=f"Claim '{claim_id}' not found.")


@app.get("/api/claims/{claim_id}/runs")
async def get_claim_runs(claim_id: str):
    """Get the agent execution runs for a claim."""
    result = _workflow_results.get(claim_id)
    if not result:
        raise HTTPException(404, detail=f"Claim '{claim_id}' not found.")

    runs = []
    step_names = {
        "patient_data": ("Patient Data Agent", "🧑‍⚕️"),
        "document_code": ("Document Code Agent", "📋"),
        "coverage_validation": ("Coverage Validation Agent", "🛡️"),
        "claim_generation": ("Claim Generation Agent", "📄"),
        "triage": ("Triage Agent", "⚖️"),
    }
    for step_key, (name, icon) in step_names.items():
        step = result.get("steps", {}).get(step_key, {})
        runs.append({
            "agent": name,
            "icon": icon,
            "step_key": step_key,
            "duration_s": step.get("duration_s", 0),
            "has_error": "error" in step,
            "error": step.get("error"),
            "output_keys": list(step.keys()) if isinstance(step, dict) else [],
            "output": step,
        })
    return {"claim_id": claim_id, "runs": runs}


@app.get("/api/claims/{claim_id}/artifacts")
async def get_claim_artifacts(claim_id: str):
    """Get artifacts (files, generated data) for a claim."""
    result = _workflow_results.get(claim_id)
    if not result:
        raise HTTPException(404, detail=f"Claim '{claim_id}' not found.")

    artifacts = []
    # PDF
    pdf_path = result.get("steps", {}).get("claim_generation", {}).get("pdf_path", "")
    if pdf_path and Path(pdf_path).exists():
        artifacts.append({
            "type": "pdf",
            "name": f"{claim_id}.pdf",
            "description": "CMS-1500 Claim Form",
            "size_bytes": Path(pdf_path).stat().st_size,
            "download_url": f"/api/claims/{claim_id}/pdf",
        })

    # Blob URL
    blob_url = result.get("steps", {}).get("claim_generation", {}).get("blob_url", "")
    if blob_url:
        artifacts.append({
            "type": "blob",
            "name": "Azure Blob",
            "description": "Cloud-stored claim PDF",
            "url": blob_url,
        })

    # JSON result
    artifacts.append({
        "type": "json",
        "name": "workflow_result.json",
        "description": "Full workflow JSON output",
        "size_bytes": len(json.dumps(result, default=str).encode()),
    })

    return {"claim_id": claim_id, "artifacts": artifacts}


@app.get("/api/claims/{claim_id}/pdf")
async def download_claim_pdf(claim_id: str):
    """Download the generated CMS-1500 PDF for a claim."""
    result = _workflow_results.get(claim_id)
    if not result:
        raise HTTPException(404, detail=f"Claim '{claim_id}' not found.")

    pdf_path = result.get("steps", {}).get("claim_generation", {}).get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(404, detail="PDF not found for this claim.")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{claim_id}.pdf",
    )


# ══════════════════════════════════════════════════════════════════
# Audit API
# ══════════════════════════════════════════════════════════════════

@app.get("/api/audit")
async def list_audit(claim_id: str | None = None):
    """List audit log entries, optionally filtered by claim_id."""
    entries = _audit_log
    if claim_id:
        entries = [e for e in entries if e.get("claim_id") == claim_id]
    return {"entries": list(reversed(entries)), "total": len(entries)}


# ══════════════════════════════════════════════════════════════════
# Knowledge Base API
# ══════════════════════════════════════════════════════════════════

@app.get("/api/knowledge")
async def list_knowledge(doc_type: str | None = None):
    """List knowledge base entries, optionally filtered by type."""
    entries = _knowledge_entries
    if doc_type:
        entries = [e for e in entries if e.get("doc_type") == doc_type]
    return {"entries": list(reversed(entries)), "total": len(entries)}


# ══════════════════════════════════════════════════════════════════
# Dashboard API
# ══════════════════════════════════════════════════════════════════

@app.get("/api/dashboard")
async def dashboard_stats():
    """Aggregate stats for the dashboard."""
    total_claims = len(_workflow_results)
    decisions = {}
    total_amount = 0.0
    for r in _workflow_results.values():
        dec = r.get("steps", {}).get("triage", {}).get("decision", "N/A")
        decisions[dec] = decisions.get(dec, 0) + 1
        total_amount += r.get("summary", {}).get("amount", 0)

    return {
        "total_claims": total_claims,
        "decisions": decisions,
        "total_amount": total_amount,
        "knowledge_entries": len(_knowledge_entries),
        "audit_entries": len(_audit_log),
        "azure_ai": settings.validate_azure_ai(),
        "blob_storage": settings.validate_blob_storage(),
        "search": settings.validate_search(),
    }


# ══════════════════════════════════════════════════════════════════
# Settings API
# ══════════════════════════════════════════════════════════════════

@app.get("/api/settings")
async def get_settings():
    """Return current application settings (masked secrets)."""
    return {
        "project_endpoint": _mask(settings.project_endpoint),
        "model_deployment_name": settings.model_deployment_name,
        "blob_container_name": settings.blob_container_name,
        "storage_connection_string": _mask(settings.storage_connection_string),
        "search_endpoint": _mask(settings.search_endpoint),
        "search_index_name": settings.search_index_name,
        "azure_ai_configured": settings.validate_azure_ai(),
        "blob_storage_configured": settings.validate_blob_storage(),
        "search_configured": settings.validate_search(),
    }


def _mask(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 12:
        return value[:3] + "***"
    return value[:8] + "***" + value[-4:]


# ══════════════════════════════════════════════════════════════════
# Chat API
# ══════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(request: Request):
    """Send a message to the MediSuite chatbot.

    Body JSON: { "message": "...", "session_id": "..." (optional) }
    """
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id", "default")

    if not message:
        raise HTTPException(400, "Message is required.")

    history = _chat_histories.get(session_id, [])
    result = chat_handler.handle_message(message, _workflow_results, history)

    # Update history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result["response"]})
    _chat_histories[session_id] = history[-20:]  # Keep last 20 turns

    _add_audit("chat_message", details=f"User: {message[:80]}")

    return {
        "response": result["response"],
        "agent": result.get("agent", "MediSuite"),
        "data": result.get("data"),
        "session_id": session_id,
    }


@app.delete("/api/chat/{session_id}")
async def clear_chat(session_id: str):
    """Clear chat history for a session."""
    _chat_histories.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


# ══════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "azure_ai": settings.validate_azure_ai(),
        "blob_storage": settings.validate_blob_storage(),
        "search": settings.validate_search(),
    }


# ── Entrypoint ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print()
    print("🏥 MediSuite Agent — Web Server")
    print("   http://localhost:8000")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)

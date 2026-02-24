"""
MediSuite Agent — Chat Handler
Routes user messages to the appropriate agent and returns responses.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from config import settings
from rag.knowledge_base import RAGKnowledgeBase

logger = logging.getLogger(__name__)


class ChatHandler:
    """Handle chat messages by routing to agents or answering from knowledge.

    Supported intents:
    - Claims lookup: "show me claim CLM-xxx", "what claims are denied"
    - Code lookup: "what is ICD-10 code J10.1", "explain CPT 99213"
    - Coverage questions: "is policy ABC123456 valid", "check coverage"
    - Triage: "why was claim X denied"
    - General: medical billing questions answered from knowledge base
    """

    def __init__(self, knowledge_base: RAGKnowledgeBase | None = None) -> None:
        self.kb = knowledge_base or RAGKnowledgeBase()
        self._icd10_db = self._load_json("icd10_codes.json")
        self._cpt4_db = self._load_json("cpt4_codes.json")
        self._policy_db = self._load_json("policy_database.json")
        self._client = None

        if settings.validate_azure_ai():
            from azure.ai.projects import AIProjectClient
            from azure.identity import DefaultAzureCredential
            self._client = AIProjectClient(
                endpoint=settings.project_endpoint,
                credential=DefaultAzureCredential(),
            )
            logger.info("ChatHandler: Azure AI client initialised.")

    @staticmethod
    def _load_json(filename: str) -> Any:
        path = settings.data_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_message(
        self,
        message: str,
        claims_store: dict[str, Any],
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Process a chat message and return a response.

        Args:
            message: The user's message text.
            claims_store: Reference to the in-memory claims dict.
            history: Optional prior conversation turns.

        Returns:
            dict with 'response' (str), 'agent' (str), 'data' (optional dict).
        """
        msg = message.strip().lower()

        # ── Try Azure AI first ────────────────────────────────────
        if self._client:
            ai_resp = self._ask_ai_chat(message, claims_store, history or [])
            if ai_resp:
                return ai_resp

        # ── Local intent routing ──────────────────────────────────

        # 1. ICD-10 code lookup
        icd_match = re.search(r'\b([A-Z]\d{2}\.?\d{0,4})\b', message, re.IGNORECASE)
        if icd_match and any(k in msg for k in ['icd', 'code', 'diagnosis', 'what is', 'explain', 'look up']):
            return self._lookup_icd10(icd_match.group(1).upper())

        # 2. CPT code lookup
        cpt_match = re.search(r'\b(\d{5})\b', message)
        if cpt_match and any(k in msg for k in ['cpt', 'procedure', 'code', 'what is', 'explain', 'cost']):
            return self._lookup_cpt4(cpt_match.group(1))

        # 3. Specific claim lookup
        claim_match = re.search(r'CLM-[A-F0-9]+', message, re.IGNORECASE)
        if claim_match:
            return self._lookup_claim(claim_match.group(0).upper(), claims_store)

        # 4. Policy / coverage check
        policy_match = re.search(r'([A-Z]{3}\d{6})', message, re.IGNORECASE)
        if policy_match and any(k in msg for k in ['policy', 'insurance', 'coverage', 'valid', 'check']):
            return self._lookup_policy(policy_match.group(1).upper())

        # 5. Aggregate claim questions
        if any(k in msg for k in ['how many', 'total', 'count', 'summary', 'stats', 'overview']):
            return self._claims_summary(claims_store)

        if any(k in msg for k in ['denied', 'deny', 'rejected']):
            return self._filter_claims_by_decision('Deny', claims_store)

        if any(k in msg for k in ['approved', 'approve']):
            return self._filter_claims_by_decision('Approve', claims_store)

        if any(k in msg for k in ['review', 'pending', 'flagged']):
            return self._filter_claims_by_decision('Review', claims_store)

        if any(k in msg for k in ['list', 'show', 'all claims']):
            return self._list_claims(claims_store)

        # 6. Help / greeting
        if any(k in msg for k in ['help', 'what can you', 'commands', 'hi', 'hello', 'hey']):
            return self._help_response()

        # 7. Agents info
        if any(k in msg for k in ['agent', 'pipeline', 'workflow', 'how does']):
            return self._agents_info()

        # 8. Fallback
        return self._fallback_response(message)

    # ------------------------------------------------------------------
    # Azure AI chat
    # ------------------------------------------------------------------
    def _ask_ai_chat(
        self,
        message: str,
        claims_store: dict[str, Any],
        history: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        """Use Azure AI for conversational responses."""
        if self._client is None:
            return None

        # Build context
        claims_summary = []
        for cid, r in claims_store.items():
            s = r.get("summary", {})
            t = r.get("steps", {}).get("triage", {})
            claims_summary.append(f"- {cid}: {s.get('patient','?')}, "
                                  f"{t.get('decision','?')}, ${s.get('amount',0)}")
        context = "\n".join(claims_summary[:10]) if claims_summary else "No claims processed yet."

        system = (
            "You are MediSuite Agent, an AI assistant for medical claim processing. "
            "You help users understand claims, look up medical codes (ICD-10, CPT-4), "
            "check insurance coverage, and explain triage decisions.\n\n"
            f"Current claims in the system:\n{context}\n\n"
            "Be concise, professional, and helpful. Use bullet points where appropriate."
        )

        try:
            with self._client:
                agent = self._client.agents.create_agent(
                    model=settings.model_deployment_name,
                    name="chat_agent",
                    instructions=system,
                )
                thread = self._client.agents.threads.create()

                # Add history
                for turn in (history or [])[-6:]:
                    self._client.agents.messages.create(
                        thread_id=thread.id,
                        role=turn.get("role", "user"),
                        content=turn.get("content", ""),
                    )

                self._client.agents.messages.create(
                    thread_id=thread.id, role="user", content=message
                )
                run = self._client.agents.runs.create_and_process(
                    thread_id=thread.id, agent_id=agent.id
                )
                if run.status == "failed":
                    return None

                messages = self._client.agents.messages.list(thread_id=thread.id)
                for m in messages:
                    if m.role == "assistant":
                        for c in m.content:
                            if hasattr(c, "text"):
                                return {
                                    "response": c.text.value,
                                    "agent": "MediSuite AI",
                                    "data": None,
                                }
        except Exception as e:
            logger.warning("ChatHandler AI error: %s", e)
        return None

    # ------------------------------------------------------------------
    # Intent handlers (local fallback)
    # ------------------------------------------------------------------
    def _lookup_icd10(self, code: str) -> dict[str, Any]:
        # Normalize: try with and without dot
        variants = [code, code.replace(".", "")]
        if len(code) > 3 and "." not in code:
            variants.append(code[:3] + "." + code[3:])

        for v in variants:
            if v in self._icd10_db:
                info = self._icd10_db[v]
                return {
                    "response": (
                        f"**ICD-10 Code: {v}**\n\n"
                        f"• **Description:** {info['description']}\n"
                        f"• **Category:** {info['category']}"
                    ),
                    "agent": "Document Code Agent",
                    "data": {"code": v, **info},
                }

        return {
            "response": f"ICD-10 code **{code}** was not found in the local database. "
                        f"Available codes include: {', '.join(list(self._icd10_db.keys())[:8])}…",
            "agent": "Document Code Agent",
            "data": None,
        }

    def _lookup_cpt4(self, code: str) -> dict[str, Any]:
        if code in self._cpt4_db:
            info = self._cpt4_db[code]
            return {
                "response": (
                    f"**CPT-4 Code: {code}**\n\n"
                    f"• **Description:** {info['description']}\n"
                    f"• **Category:** {info['category']}\n"
                    f"• **Average Cost:** ${info.get('average_cost', 0):,.2f}"
                ),
                "agent": "Document Code Agent",
                "data": {"code": code, **info},
            }
        return {
            "response": f"CPT-4 code **{code}** was not found. "
                        f"Available codes: {', '.join(list(self._cpt4_db.keys())[:8])}…",
            "agent": "Document Code Agent",
            "data": None,
        }

    def _lookup_claim(self, claim_id: str, claims_store: dict) -> dict[str, Any]:
        result = claims_store.get(claim_id)
        if not result:
            return {
                "response": f"Claim **{claim_id}** not found. "
                            f"Active claims: {', '.join(claims_store.keys()) or 'none'}.",
                "agent": "Triage Agent",
                "data": None,
            }

        s = result.get("summary", {})
        steps = result.get("steps", {})
        triage = steps.get("triage", {})
        cov = steps.get("coverage_validation", {})
        doc = steps.get("document_code", {})

        response = (
            f"**Claim {claim_id}**\n\n"
            f"• **Patient:** {s.get('patient', '?')}\n"
            f"• **Decision:** {triage.get('decision', '?')}\n"
            f"• **Justification:** {triage.get('justification', 'N/A')}\n"
            f"• **Amount:** ${s.get('amount', 0):,.2f}\n"
            f"• **Validation:** {cov.get('validation_status', '?')}\n"
            f"• **ICD-10 Codes:** {', '.join(doc.get('icd10_codes', []))}\n"
            f"• **CPT-4 Codes:** {', '.join(doc.get('cpt4_codes', []))}\n"
            f"• **Confidence:** {triage.get('confidence', 0) * 100:.0f}%"
        )
        return {
            "response": response,
            "agent": "Triage Agent",
            "data": {"claim_id": claim_id, "decision": triage.get("decision")},
        }

    def _lookup_policy(self, policy_number: str) -> dict[str, Any]:
        policies = self._policy_db if isinstance(self._policy_db, list) else []
        for p in policies:
            if p.get("policy_number") == policy_number:
                status = p.get("coverage", "Unknown")
                return {
                    "response": (
                        f"**Policy {policy_number}**\n\n"
                        f"• **Provider:** {p.get('provider')}\n"
                        f"• **Status:** {status}\n"
                        f"• **Plan Type:** {p.get('plan_type')}\n"
                        f"• **Effective:** {p.get('effective_date')} → {p.get('expiry_date')}\n"
                        f"• **Copay:** ${p.get('copay', 0):,.2f}\n"
                        f"• **Deductible:** ${p.get('deductible', 0):,.2f} "
                        f"(met: ${p.get('deductible_met', 0):,.2f})\n"
                        f"• **Covered Services:** {', '.join(p.get('covered_services', []))}"
                    ),
                    "agent": "Coverage Validation Agent",
                    "data": p,
                }
        return {
            "response": f"Policy **{policy_number}** not found in the database.",
            "agent": "Coverage Validation Agent",
            "data": None,
        }

    def _claims_summary(self, claims_store: dict) -> dict[str, Any]:
        total = len(claims_store)
        if total == 0:
            return {
                "response": "No claims have been processed yet. Submit a new claim to get started!",
                "agent": "Dashboard",
                "data": None,
            }

        decisions = {}
        total_amount = 0
        for r in claims_store.values():
            dec = r.get("steps", {}).get("triage", {}).get("decision", "Unknown")
            decisions[dec] = decisions.get(dec, 0) + 1
            total_amount += r.get("summary", {}).get("amount", 0)

        lines = [f"**Claims Summary**\n"]
        lines.append(f"• **Total Claims:** {total}")
        for dec, count in sorted(decisions.items()):
            lines.append(f"• **{dec}:** {count}")
        lines.append(f"• **Total Amount:** ${total_amount:,.2f}")

        return {"response": "\n".join(lines), "agent": "Dashboard", "data": decisions}

    def _filter_claims_by_decision(self, decision: str, claims_store: dict) -> dict[str, Any]:
        matches = []
        for cid, r in claims_store.items():
            dec = r.get("steps", {}).get("triage", {}).get("decision", "")
            if dec == decision:
                s = r.get("summary", {})
                matches.append(f"• **{cid}** — {s.get('patient', '?')} — ${s.get('amount', 0):,.2f}")

        if not matches:
            return {
                "response": f"No claims with decision **{decision}** found.",
                "agent": "Triage Agent",
                "data": None,
            }

        return {
            "response": f"**{decision} Claims ({len(matches)}):**\n\n" + "\n".join(matches),
            "agent": "Triage Agent",
            "data": {"decision": decision, "count": len(matches)},
        }

    def _list_claims(self, claims_store: dict) -> dict[str, Any]:
        if not claims_store:
            return {
                "response": "No claims in the system yet.",
                "agent": "Dashboard",
                "data": None,
            }
        lines = ["**All Claims:**\n"]
        for cid, r in claims_store.items():
            s = r.get("summary", {})
            dec = r.get("steps", {}).get("triage", {}).get("decision", "?")
            lines.append(f"• **{cid}** — {s.get('patient', '?')} — {dec} — ${s.get('amount', 0):,.2f}")
        return {"response": "\n".join(lines), "agent": "Dashboard", "data": None}

    def _help_response(self) -> dict[str, Any]:
        return {
            "response": (
                "👋 **Hi! I'm the MediSuite Agent chatbot.** Here's what I can help with:\n\n"
                "**Claims**\n"
                "• \"Show me claim CLM-XXXXXXXX\"\n"
                "• \"List all claims\" / \"Show denied claims\"\n"
                "• \"How many claims are approved?\"\n\n"
                "**Medical Codes**\n"
                "• \"What is ICD-10 code J10.1?\"\n"
                "• \"Explain CPT code 99213\"\n\n"
                "**Insurance**\n"
                "• \"Check policy ABC123456\"\n"
                "• \"Is policy DEF789012 valid?\"\n\n"
                "**System**\n"
                "• \"Show me a summary\" / \"Stats\"\n"
                "• \"How does the pipeline work?\"\n"
                "• \"What agents are available?\""
            ),
            "agent": "MediSuite",
            "data": None,
        }

    def _agents_info(self) -> dict[str, Any]:
        return {
            "response": (
                "**MediSuite Agent Pipeline**\n\n"
                "The system uses 5 specialized AI agents in sequence:\n\n"
                "1. 🧑‍⚕️ **Patient Data Agent** — Validates patient info, extracts diagnoses from clinical notes\n"
                "2. 📋 **Document Code Agent** — Looks up ICD-10 and CPT-4 codes, calculates charges\n"
                "3. 🛡️ **Coverage Validation Agent** — Checks insurance policy validity and coverage\n"
                "4. 📄 **Claim Generation Agent** — Creates CMS-1500 PDF forms\n"
                "5. ⚖️ **Triage Agent** — Makes approve/deny/review decisions with risk assessment\n\n"
                "Each agent passes its output to the next, building a complete claim record."
            ),
            "agent": "MediSuite",
            "data": None,
        }

    def _fallback_response(self, message: str) -> dict[str, Any]:
        return {
            "response": (
                f"I'm not quite sure how to help with that. Try asking me about:\n\n"
                f"• A specific claim (e.g., \"Tell me about CLM-A602CF43\")\n"
                f"• Medical codes (e.g., \"What is ICD-10 J10.1?\")\n"
                f"• Insurance policies (e.g., \"Check policy ABC123456\")\n"
                f"• Claims summary (e.g., \"Show me all denied claims\")\n\n"
                f"Type **help** for the full list of commands."
            ),
            "agent": "MediSuite",
            "data": None,
        }

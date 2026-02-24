"""
MediSuite Agent — Base Agent
Abstract base class that manages the Azure AI Foundry Agent lifecycle.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from config import settings
from rag.knowledge_base import RAGKnowledgeBase

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for every MediSuite agent.

    Subclasses must implement ``run()``.  The base class provides helpers
    for interacting with the Azure AI Foundry Agent Service and the
    shared RAG Knowledge Base.
    """

    agent_name: str = "base_agent"
    agent_instructions: str = "You are a helpful assistant."

    def __init__(self, knowledge_base: RAGKnowledgeBase | None = None) -> None:
        self.kb = knowledge_base or RAGKnowledgeBase()
        self._client: AIProjectClient | None = None

        if settings.validate_azure_ai():
            self._client = AIProjectClient(
                endpoint=settings.project_endpoint,
                credential=DefaultAzureCredential(),
            )
            logger.info("%s: Azure AI client initialised.", self.agent_name)
        else:
            logger.warning(
                "%s: Azure AI not configured — agent will use local-only logic.",
                self.agent_name,
            )

    # ------------------------------------------------------------------
    # Azure AI helper — send a prompt and get structured JSON back
    # ------------------------------------------------------------------
    def _ask_ai(self, prompt: str, system_instructions: str | None = None) -> str:
        """Send a prompt to the Azure AI agent and return the response text.

        Falls back to an empty string when Azure is not available.
        """
        if self._client is None:
            logger.info("[AI-MOCK] %s._ask_ai() — no Azure client", self.agent_name)
            return ""

        instructions = system_instructions or self.agent_instructions

        with self._client:
            agent = self._client.agents.create_agent(
                model=settings.model_deployment_name,
                name=self.agent_name,
                instructions=instructions,
            )
            thread = self._client.agents.threads.create()
            self._client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt,
            )
            run = self._client.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent.id,
            )

            if run.status == "failed":
                logger.error(
                    "%s: AI run failed — %s", self.agent_name, run.last_error
                )
                return ""

            messages = self._client.agents.messages.list(thread_id=thread.id)
            # The last assistant message contains the answer
            for msg in messages:
                if msg.role == "assistant":
                    for content in msg.content:
                        if hasattr(content, "text"):
                            return content.text.value
            return ""

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Best-effort extraction of JSON from an AI response string."""
        text = text.strip()
        # Try to find JSON block within markdown fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "%s: failed to parse JSON from AI response: %s",
                self.agent_name,
                text[:200],
            )
            return {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's primary task.

        Args:
            input_data: Agent-specific input dictionary.

        Returns:
            Agent-specific output dictionary.
        """
        ...

from __future__ import annotations

import logging
from typing import Optional

from llm.base_llm import BaseLLM, GenerationConfig
from llm.ollama_backend import OllamaBackend
from llm.groq_backend import GroqBackend
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes an agent role to the correct LLM backend, falling back to Groq if Ollama is down."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cache: dict[str, BaseLLM] = {}

    def _ollama_model_for_role(self, role: str) -> str:
        mapping = {
            "planner": self._settings.planner_model,
            "coder": self._settings.coder_model,
            "reviewer": self._settings.reviewer_model,
            "executor": self._settings.reviewer_model,
        }
        return mapping.get(role, self._settings.planner_model)

    async def get_backend(self, role: str, prefer_local: bool = True) -> BaseLLM:
        cache_key = f"{role}:{'local' if prefer_local else 'cloud'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        model_name = self._ollama_model_for_role(role)

        if prefer_local:
            ollama = OllamaBackend(
                model_name=model_name,
                base_url=self._settings.ollama_base_url,
            )
            if await ollama.health_check():
                logger.info("Router: using Ollama %r for role %r", model_name, role)
                self._cache[cache_key] = ollama
                return ollama
            logger.warning(
                "Ollama backend unhealthy for role %r — falling back to Groq", role
            )

        if not self._settings.groq_api_key:
            raise RuntimeError(
                "Ollama is unavailable and GROQ_API_KEY is not set. "
                "Either start Ollama or set GROQ_API_KEY in .env."
            )

        groq = GroqBackend(
            ollama_model_name=model_name,
            api_key=self._settings.groq_api_key,
        )
        logger.info("Router: using Groq fallback for role %r", role)
        self._cache[cache_key] = groq
        return groq

    async def close_all(self) -> None:
        for backend in self._cache.values():
            await backend.close()
        self._cache.clear()


_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router

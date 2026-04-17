from __future__ import annotations

import logging
import os
import time
import json
import uuid
import asyncio
from typing import AsyncIterator, Optional

import httpx

from llm.base_llm import BaseLLM, GenerationConfig, LLMResponse

logger = logging.getLogger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MODEL_MAP: dict[str, str] = {
    "qwen2.5:14b":        "llama-3.1-70b-versatile",
    "qwen2.5-coder:32b":  "llama-3.1-70b-versatile",
    "mistral:7b":         "gemma2-9b-it",
}

_DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"


class GroqBackend(BaseLLM):

    def __init__(
        self,
        ollama_model_name: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        groq_model = MODEL_MAP.get(ollama_model_name)
        if groq_model is None:
            groq_model = _DEFAULT_GROQ_MODEL
            logger.warning(
                "No Groq mapping for %r — using default %r",
                ollama_model_name, _DEFAULT_GROQ_MODEL,
            )

        super().__init__(model_name=groq_model, timeout=timeout)

        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Groq API key not provided. Set GROQ_API_KEY in .env "
                "or pass api_key= to GroqBackend()."
            )

        self._client = httpx.AsyncClient(
            base_url=_GROQ_BASE_URL,
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=5.0),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        request_id = str(uuid.uuid4())
        cfg = config or GenerationConfig()
        start = time.monotonic()

        logger.debug("Groq request started | id=%s | model=%s", request_id, self.model_name)

        response = await self._with_retry(self._raw_generate, messages, cfg)
        response.latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Groq response | id=%s | tokens=%d | latency=%.0fms",
            request_id, response.tokens_used, response.latency_ms,
        )
        return response

    async def stream(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        request_id = str(uuid.uuid4())
        cfg = config or GenerationConfig()
        payload = self._build_payload(messages, cfg, stream=True)

        logger.debug("Groq stream started | id=%s", request_id)

        try:
            stream_ctx = self._client.stream(
                "POST", "/chat/completions", json=payload
            )

            async with await asyncio.wait_for(stream_ctx.__aenter__(), timeout=self.timeout) as resp:
                resp.raise_for_status()

                async for raw_line in resp.aiter_lines():
                    raw_line = raw_line.strip()

                    if not raw_line or not raw_line.startswith("data: "):
                        continue

                    data_part = raw_line[len("data: "):]
                    if data_part == "[DONE]":
                        logger.debug("Groq stream finished | id=%s", request_id)
                        return

                    try:
                        obj = json.loads(data_part)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        chunk = delta.get("content", "")
                        if chunk:
                            yield chunk
                    except Exception:
                        continue

        except asyncio.TimeoutError:
            logger.error("Groq stream timeout | id=%s | timeout=%ss", request_id, self.timeout)
            yield ""
        except httpx.HTTPStatusError as exc:
            logger.error("Groq stream HTTP error | id=%s | %s", request_id, exc)
            yield ""
        except Exception as exc:
            logger.error("Groq stream unexpected error | id=%s | %s", request_id, exc)
            yield ""

    async def health_check(self) -> bool:
        if self._healthy is not None:
            return self._healthy

        try:
            resp = await self._client.get("/models", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()

            available = [m["id"] for m in data.get("data", [])]

            if self.model_name in available:
                self._healthy = True
                logger.debug("Groq health OK: model %r available", self.model_name)
                return True

            self._healthy = False
            logger.warning(
                "Groq health: model %r not in available list: %s",
                self.model_name, available[:5],
            )
            return False

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                logger.error(
                    "Groq API key is invalid or expired. "
                    "Check GROQ_API_KEY in your .env."
                )
            self._healthy = False
            return False

        except Exception as exc:
            logger.warning("Groq health check failed: %s", exc)
            self._healthy = False
            return False

    async def __aenter__(self) -> "GroqBackend":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _raw_generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> LLMResponse:
        payload = self._build_payload(messages, config, stream=False)

        try:
            resp = await asyncio.wait_for(
                self._client.post("/chat/completions", json=payload),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error("Groq request timeout after %ss", self.timeout)
            raise RuntimeError(f"Groq request timeout after {self.timeout}s")
        except httpx.ReadTimeout:
            raise RuntimeError(
                f"Groq timed out after {self.timeout}s for model {self.model_name}. "
                "Consider using a smaller/faster Groq model."
            )

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "5"))
            logger.warning(
                "Groq rate limit hit. Waiting %.1fs as instructed by API.", retry_after
            )
            await asyncio.sleep(retry_after)
            resp = await asyncio.wait_for(
                self._client.post("/chat/completions", json=payload),
                timeout=self.timeout
            )

        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        text = choice["message"]["content"].strip()
        finish_reason = choice.get("finish_reason", "stop")

        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)

        return LLMResponse(
            text=text,
            model=self.model_name,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            raw=data,
        )

    def _build_payload(
        self,
        messages: list[dict],
        config: GenerationConfig,
        stream: bool,
    ) -> dict:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "frequency_penalty": max(0.0, config.repeat_penalty - 1.0),
            "stream": stream,
        }

        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        return payload
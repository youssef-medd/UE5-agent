from __future__ import annotations
import json
import logging
import time
from typing import AsyncIterator, Optional
import httpx
from llm.base_llm import BaseLLM, GenerationConfig, LLMResponse
logger = logging.getLogger(__name__)
_DEFAULT_BASE_URL = "http://localhost:11434"
class OllamaBackend(BaseLLM):
    def __init__(
        self,
        model_name: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 90.0,
    ):
        super().__init__(model_name=model_name, timeout=timeout)
        self.base_url = base_url.rstrip("/") 
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=10.0,    
                read=timeout,    
                write=30.0,      
                pool=5.0,        
            ),
            headers={"Content-Type": "application/json"},
        )
    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        cfg = config or GenerationConfig()
        start = time.monotonic()
        response = await self._with_retry(self._raw_generate, messages, cfg)
        response.latency_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "Ollama generate: model=%s, tokens=%d, latency=%.0fms",
            self.model_name, response.tokens_used, response.latency_ms,
        )
        return response
 
    async def stream(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        cfg = config or GenerationConfig()
        payload = self._build_payload(messages, cfg, stream=True)
 
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
 
                async for raw_line in resp.aiter_lines():
                    if not raw_line.strip():
                        continue 
                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError:
                        logger.warning("Ollama stream: unparseable line: %r", raw_line)
                        continue
                    chunk = obj.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if obj.get("done", False):
                        used = obj.get("eval_count", 0) + obj.get("prompt_eval_count", 0)
                        logger.debug(
                            "Ollama stream complete: model=%s, tokens=%d",
                            self.model_name, used,
                        )
                        return
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama stream HTTP error: %s", exc)
            yield ""  
        except Exception as exc:
            logger.error("Ollama stream unexpected error: %s", exc)
            yield ""
    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            available = [m["name"] for m in data.get("models", [])]
            if self.model_name in available:
                self._healthy = True
                logger.debug("Ollama health OK: model %r is available", self.model_name)
                return True
            else:
                self._healthy = False
                logger.warning(
                    "Ollama health FAIL: model %r not found. Available: %s",
                    self.model_name, available,
                )
                return False
        except Exception as exc:
            self._healthy = False
            logger.warning("Ollama health check failed: %s", exc)
            return False
    async def __aenter__(self) -> "OllamaBackend":
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
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise RuntimeError(
                    f"Model {self.model_name!r} not found in Ollama. "
                    f"Run: ollama pull {self.model_name}"
                ) from exc
            raise 
        data = resp.json()
        text = data.get("message", {}).get("content", "").strip()
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        finish_reason = data.get("done_reason", "stop")
        return LLMResponse(
            text=text,
            model=self.model_name,
            tokens_used=prompt_tokens + completion_tokens,
            finish_reason=finish_reason,
            raw=data,
        )
    def _build_payload(
        self,
        messages: list[dict],
        config: GenerationConfig,
        stream: bool,
    ) -> dict:
        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repeat_penalty,
            },
        }
        if config.stop_sequences:
            payload["options"]["stop"] = config.stop_sequences 
        return payload
 

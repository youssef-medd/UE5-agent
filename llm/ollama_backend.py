from __future__ import annotations
import json
import logging
import time
from typing import AsyncIterator , Optional
import httpx
from llm.base_llm import BaseLLM , GenerationConfig , LLMResponse
logger = logging.getLogger(__name__)
_DEFAULT_BASE_URL = "http://localhost:11434"
class OllamaBackend(BaseLLM) :
    def __init__(
            self ,
            model_name : str ,
            base_url : str = _DEFAULT_BASE_URL ,
            timeout : float = 90.0 ,
    ):
        super().__init__(model_name=model_name , timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url = self.base_url ,
            timeout = httpx.Timeout(
                connect = 10.0
                read = timeout ,
                write = 30.0,
                pool = 5.0
            )
            headers = {"Content_Type" : "application"}
        )
    async def generate(
            self ,
            message : list[dict],
            config : Optional[GenerationConfig] = None ,
    ) -> LLMResponse :
        cfg = config or GenerationConfig()
        start = time.monotonic()
        response = await self._with_retry(self._raw_generate , messages ,cfg)
        response.latency_ms = (time.monotonic() - start) *1000
        logger.debug(
            "ollama genrate : model=%s , tokens = %d , latency = %.0fms",
            self.model_name , response.tokens_used , response.latency_ms ,
        )
        return response
    async def stream(self, messages : list[dict], config = None):
        return await super().stream(messages, config)
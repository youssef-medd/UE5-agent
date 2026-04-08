from __future__ import annotations
 
import abc
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
logger = logging.getLogger(__name__)
@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)
 
    def is_complete(self) -> bool:
        return self.finish_reason == "stop"
    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"LLMResponse(model={self.model!r}, tokens={self.tokens_used}, "
            f"latency={self.latency_ms:.0f}ms, text={preview!r}...)"
        )
@dataclass
class GenerationConfig:
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False
    system_prompt: str = ""
    @classmethod
    def for_role(cls, role: str) -> "GenerationConfig":
        presets = {
            "planner": cls(temperature=0.5, max_tokens=1024),
            "coder":   cls(temperature=0.1, max_tokens=4096,
                           stop_sequences=["```\n\n", "# END"]),
            "reviewer": cls(temperature=0.3, max_tokens=1024),
            "executor": cls(temperature=0.0, max_tokens=256),
        }
        if role not in presets:
            raise ValueError(
                f"Unknown role {role!r}. Valid roles: {list(presets)}"
            )
        return presets[role]
class RetryMixin:
    max_attempts = 4
    base_delay : float = 0.1
    max_delay : float = 30.0
    jitter : float = 0.5
    async def _with_retry(self , coro_fn , *args , **kwargs):
        import random
        last_exc : Optional[Exception] = None
        for attempt in range(self.max_attempts) :
            try :
                return await coro_fn(*args , **kwargs)
            except Exception as e :
                last_exc = e
                if attempt == self.max_attempts - 1 :
                    break
                delay = min(
                    self.base_delay * (2 ** attempt) + random.uniform(0 , self.jitter),
                    self.max_delay
                )
                logger.warning(
                    attempt + self.max_attempts , e , delay ,
                )
                await asyncio.sleep(delay)
        logger.error("all %d retry attempts exhausted" , self.max_attempts)
        raise last_exc
class TokenBudget :
    def __init__(self , limit : int = 20000):
        self.limit = limit
        self._calls : list[dict] = []
    def record(self , response : LLMResponse) -> None :
        self._calls.append({
            "model" : response.model ,
            "tokens" : response.tokens_used,
            "latency_ms" : response.latency_ms,
            "timestamp" : time.time(),
        })
    @property
    def total_tokens(self) -> int :
        return sum(c["tokens"] for c in self._calls)
    @property
    def total_calls(self) -> int :
        return len(self._calls)
    def assert_within_limit(self) -> None :
        if self.total_tokens > self.limit :
            raise RuntimeError(
                f"Token budget exceeded: {self.total_tokens} / {self.limit} used "
                f"across {self.total_calls} calls."
            )
    def summary(self) -> str :
        avg_lat = (
            sum(c["latency_ms"] for c in self._calls) / len(self._calls)
            if self.self._calls else 0
        )
        return (
            f"TokenBudget: {self.total_tokens}/{self.limit} tokens used "
            f"in {self.total_calls} calls (avg latency {avg_lat:.0f}ms)"
        )

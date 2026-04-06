from __future__ import annotations
import abc
import asyncio
import time
from dataclasses import dataclass , field
from typing import AsyncGenerator , Optional
logger = logging.getLogger(__name__)
@dataclass
class LLMResponse :
    text : str
    model : str 
    tokens_used : int = 0
    latency_ms : float = 0.0
    finish_reason : str = "stop"
    raw : dict = field(default_factory= dict)
    def is_complete(self) -> bool :
        return self.finish_reason == "stop"
    def __repr__(self) -> bool :
        preview = self.text[:80].replace("\n" , " ")
        return (
            f"LLMResponse(model={self.model!r} , tokens = {self.tokens_used} , "
            f"latency = {self.latency_ms : .0f}ms , text = {preview!r}...)"
        )
    max_tokens : int = 2048
    temperature : float = 0.2
    top_p : float = 0.9
    top_k : int = 40
    repeat_penalty : float = 1.1
    stop_sequences : list[str] = field(default_factory=list)
    stream : bool = False
    system_prompt : str = ""
    @classmethod
    def for_role(cls , role : str) -> "GenerationConfig" :
        presets = {
            "planner": cls(temperature=0.5, max_tokens=1024),
            "coder":   cls(temperature=0.1, max_tokens=4096,
                           stop_sequences=["```\n\n", "# END"]),
            "reviewer": cls(temperature=0.3, max_tokens=1024),
            "executor": cls(temperature=0.0, max_tokens=256),
        }
        if role not in presets :
            raise ValueError(
                f"unknown role {role!r}. valid roles : {list(presets)}"
            )
        return presets[role]

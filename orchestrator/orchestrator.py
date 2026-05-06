from __future__ import annotations

import logging
import uuid

from agents.planner_agent import PlannerAgent
from agents.coder_agent import CoderAgent
from agents.reviewer_agent import ReviewerAgent
from agents.executor_agent import ExecutorAgent
from memory.short_term import ShortTermMemory
from memory.rag_retriever import RAGRetriever
from memory.task_history import TaskHistory, TaskRecord
from sandbox.code_validator import CodeValidator
from orchestrator.task_parser import parse_task
from config.settings import get_settings
from events.bus import get_bus
from events.event_types import Event, EventKind

logger = logging.getLogger(__name__)


class Orchestrator:
    """Routes a user prompt through the full agent pipeline."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._history = TaskHistory()
        self._rag = RAGRetriever()
        self._validator = CodeValidator()
        self._bus = get_bus()
        self._memory = ShortTermMemory()

    async def run(self, prompt: str, dry_run: bool | None = None) -> dict:
        task_id = str(uuid.uuid4())[:8]
        effective_dry_run = dry_run if dry_run is not None else self._settings.dry_run

        parsed = parse_task(prompt)
        if "dry_run" in parsed.flags:
            effective_dry_run = True

        logger.info("Orchestrator: task=%s intent=%s dry_run=%s", task_id, parsed.intent, effective_dry_run)

        await self._bus.publish(Event(
            kind=EventKind.TASK_CREATED,
            task_id=task_id,
            payload={"prompt": prompt, "intent": parsed.intent},
        ))

        record = TaskRecord(task_id=task_id, prompt=prompt)
        self._memory.add_turn("user", prompt)

        try:
            result = await self._pipeline(task_id, prompt, effective_dry_run, record)
            record.success = result.get("success", False)
            record.final_code = result.get("code", "")

            await self._bus.publish(Event(
                kind=EventKind.TASK_COMPLETED if record.success else EventKind.TASK_FAILED,
                task_id=task_id,
            ))
        except Exception as exc:
            logger.error("Orchestrator pipeline error: %s", exc)
            record.error = str(exc)
            result = {"success": False, "error": str(exc)}
            await self._bus.publish(Event(kind=EventKind.TASK_FAILED, task_id=task_id))
        finally:
            import time
            record.completed_at = time.time()
            self._history.save(record)

        return result

    async def _pipeline(self, task_id: str, prompt: str, dry_run: bool, record: TaskRecord) -> dict:
        # 1. Plan
        planner = PlannerAgent(task_id=task_id)
        plan_result = await planner.run(prompt)
        steps = plan_result.output or [{"step": 1, "action": prompt, "agent": "coder", "details": ""}]
        record.steps = steps

        final_code = ""
        last_review = {}

        for step in steps:
            rag_context = self._rag.retrieve(step.get("action", ""), n_results=2)

            # 2. Code
            coder = CoderAgent(task_id=task_id)
            code_result = await coder.run(
                prompt,
                context={"plan_step": step, "rag_context": rag_context},
            )
            if not code_result.success or not code_result.output:
                continue

            code = code_result.output

            # 2b. Static validation
            validation = self._validator.validate(code)
            if not validation.valid:
                logger.warning("Step %s: static validation failed: %s", step.get("step"), validation.errors)
                continue

            # 3. Review
            reviewer = ReviewerAgent(task_id=task_id)
            review_result = await reviewer.run(prompt, context={"code": code})
            last_review = review_result.output or {}

            if not last_review.get("approved", False) and last_review.get("risk_level") == "high":
                logger.warning("Step %s: reviewer blocked execution", step.get("step"))
                continue

            final_code = code

            # 4. Execute
            executor = ExecutorAgent(task_id=task_id)
            exec_result = await executor.run(
                prompt,
                context={"code": code, "review": last_review, "dry_run": dry_run},
            )

            if exec_result.success:
                self._memory.add_agent_result("executor", exec_result.output, True)

        return {
            "success": bool(final_code),
            "code": final_code,
            "review": last_review,
            "task_id": task_id,
            "dry_run": dry_run,
        }

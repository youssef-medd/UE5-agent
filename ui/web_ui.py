from __future__ import annotations

import asyncio
import logging

from orchestrator.orchestrator import Orchestrator
from config.settings import get_settings

logger = logging.getLogger(__name__)

_orchestrator: Orchestrator | None = None


def _get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def _run_prompt(prompt: str, dry_run: bool) -> tuple[str, str, str]:
    if not prompt.strip():
        return "", "", ""

    async def _inner():
        return await _get_orchestrator().run(prompt, dry_run=dry_run)

    result = asyncio.run(_inner())
    code = result.get("code", "")
    review = result.get("review", {})
    status = "Success" if result.get("success") else f"Failed: {result.get('error', '')}"
    review_text = (
        f"Approved: {review.get('approved', False)} | Risk: {review.get('risk_level', '?')}\n"
        f"{review.get('summary', '')}"
        if review else ""
    )
    return code, review_text, status


def launch(share: bool = False) -> None:
    try:
        import gradio as gr
    except ImportError:
        raise RuntimeError("Gradio not installed. Run: pip install gradio")

    settings = get_settings()

    with gr.Blocks(title="UE5 Agent") as demo:
        gr.Markdown("# UE5 Agent\nTurn plain English into UE5 Blueprint or Python code.")

        with gr.Row():
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="e.g. add a point light above the player spawn",
                lines=3,
            )

        with gr.Row():
            dry_run_check = gr.Checkbox(label="Dry Run (no UE5 writes)", value=settings.dry_run)
            run_btn = gr.Button("Run", variant="primary")

        with gr.Row():
            code_out = gr.Code(label="Generated Code", language="python")

        with gr.Row():
            review_out = gr.Textbox(label="Review", lines=3)
            status_out = gr.Textbox(label="Status")

        run_btn.click(
            fn=_run_prompt,
            inputs=[prompt_box, dry_run_check],
            outputs=[code_out, review_out, status_out],
        )

    demo.launch(share=share)

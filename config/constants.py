from __future__ import annotations

AGENT_ROLES = ("planner", "coder", "reviewer", "executor")

DEFAULT_PLANNER_MODEL = "qwen2.5:14b"
DEFAULT_CODER_MODEL = "qwen2.5-coder:32b"
DEFAULT_REVIEWER_MODEL = "mistral:7b"

UE5_REMOTE_CONTROL_PORT = 30010
UE5_REMOTE_CONTROL_TIMEOUT = 15.0

TOKEN_BUDGET_DEFAULT = 50_000
MAX_PLAN_STEPS = 10
MAX_CODE_RETRIES = 3

LOG_DIR = "logs"
AGENT_RUN_LOG_DIR = "logs/agent_runs"
ERROR_LOG_DIR = "logs/errors"
KNOWLEDGE_DIR = "knowledge"
CHROMA_PERSIST_DIR = ".chromadb"

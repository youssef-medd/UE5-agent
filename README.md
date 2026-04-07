# ue5-agent

> A multi-agent AI system that understands, plans, writes, and executes code directly inside Unreal Engine 5 — using local LLMs via Ollama with Groq as a cloud fallback.

---

## What this is

`ue5-agent` turns a plain English prompt like *"add a gravity gun mechanic"* into working Blueprint or Python code running live inside your UE5 project. It does this by routing your request through a pipeline of specialized AI agents — each one built for a specific job — rather than relying on a single model to do everything.

No cloud dependency by default. Everything runs on your machine using Ollama. Groq kicks in automatically if your local GPU is overloaded.

---

## Architecture overview

```
User prompt
    │
    ▼
┌─────────────────────────────────────┐
│           Orchestrator              │  Routes task → correct agent(s)
│  task_parser · task_queue · state   │  Manages shared Redis/JSON state
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┬────────────┐
    ▼          ▼          ▼            ▼
 Planner    Coder     Reviewer     Executor
 Agent      Agent     Agent        Agent
 (plan)    (write)  (validate)   (run in UE5)
    │          │          │            │
    └──────────┴──────────┴────────────┘
                          │
               ┌──────────┴──────────┐
               │      LLM Layer      │
               │  Ollama · Groq      │
               │  model_router       │
               └──────────┬──────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      Tool Layer       Memory           Sandbox
  UE5 Python API     ChromaDB        AST validator
  Remote Control     RAG retriever   Dry-run mode
  Blueprint writer   Task history    Rollback
```

---

## Agent roles

| Agent | Model | Job |
|---|---|---|
| **Planner** | Qwen2.5-14B | Breaks your prompt into ordered subtasks |
| **Coder** | Qwen2.5-Coder-32B | Writes UE5 Python / Blueprint code |
| **Reviewer** | Mistral-7B | Validates code, checks safety, flags risks |
| **Executor** | Deterministic | Sends validated code to UE5 via HTTP |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Use pyenv or conda |
| Ollama | latest | [ollama.com](https://ollama.com) |
| Unreal Engine | 5.3+ | Remote Control Plugin must be enabled |
| Redis | 7+ | Used for shared agent state |
| ChromaDB | via pip | Auto-initialized on first run |

Optional:
- **Groq API key** — free tier is enough for fallback use. Get one at [console.groq.com](https://console.groq.com)

---

## Installation

### 1. Clone and set up environment

```bash
git clone https://github.com/yourname/ue5-agent.git
cd ue5-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Pull local models via Ollama

```bash
# Planning agent
ollama pull qwen2.5:14b

# Coding agent (large — ~20 GB)
ollama pull qwen2.5-coder:32b

# Reviewer / fallback
ollama pull mistral:7b
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# --- LLM backends ---
OLLAMA_BASE_URL=http://localhost:11434
GROQ_API_KEY=gsk_...              # optional, enables cloud fallback

# --- Model assignments ---
PLANNER_MODEL=qwen2.5:14b
CODER_MODEL=qwen2.5-coder:32b
REVIEWER_MODEL=mistral:7b

# --- UE5 connection ---
UE5_REMOTE_CONTROL_HOST=localhost
UE5_REMOTE_CONTROL_PORT=30010

# --- State backend ---
REDIS_URL=redis://localhost:6379/0

# --- Behavior ---
DRY_RUN=false                     # set true to simulate without touching UE5
LOG_LEVEL=INFO
```

### 4. Enable UE5 Remote Control Plugin

In Unreal Editor:
1. Edit → Plugins → search "Remote Control API" → enable it
2. Edit → Project Settings → Plugins → Remote Control → set HTTP Server Port to `30010`
3. Restart the editor

### 5. Initialize the knowledge base

```bash
python scripts/build_knowledge_base.py
```

This indexes all UE5 Python API docs, Blueprint patterns, and class references into ChromaDB for RAG retrieval at runtime.

---

## Running

```bash
# Interactive terminal UI
python main.py --ui cli

# Web dashboard (Gradio)
python main.py --ui web

# Single prompt, headless
python main.py --prompt "add a point light above the player spawn"

# Dry-run (no UE5 writes)
python main.py --prompt "spawn 10 crates in a grid" --dry-run
```

---

## Project structure

```
ue5-agent/
├── main.py                    # Entry point
├── .env                       # Model paths, ports, API keys
├── requirements.txt
│
├── orchestrator/              # Task routing + state management
│   ├── orchestrator.py        # Routes prompt → agent pipeline
│   ├── task_parser.py         # Extracts intent + parameters
│   ├── task_queue.py          # Priority queue for multi-step tasks
│   └── state_manager.py       # Shared Redis / JSON state
│
├── agents/                    # Specialized AI agents
│   ├── base_agent.py          # Abstract class all agents extend
│   ├── planner_agent.py       # Qwen2.5-14B — decomposes tasks
│   ├── coder_agent.py         # Qwen2.5-Coder-32B — writes code
│   ├── reviewer_agent.py      # Mistral-7B — validates + safety
│   └── executor_agent.py      # Deterministic — calls UE5 tools
│
├── llm/                       # ★ Model abstraction layer (built Day 1)
│   ├── base_llm.py            # Abstract LLM interface
│   ├── ollama_backend.py      # Local Ollama async calls
│   ├── groq_backend.py        # Groq cloud fallback
│   ├── model_router.py        # Maps agent role → model → backend
│   └── prompt_templates.py    # Typed system prompts per agent
│
├── tools/                     # UE5 tool wrappers
│   ├── tool_registry.py       # Registers + validates tool calls
│   ├── ue5_python_bridge.py   # Spawn / delete / move actors
│   ├── remote_control.py      # HTTP → UE5 port 30010
│   ├── blueprint_writer.py    # Generates + compiles Blueprints
│   ├── log_watcher.py         # Tails UE5 output log live
│   ├── asset_manager.py       # Imports meshes, materials
│   └── world_query.py         # Reads current world state
│
├── memory/                    # Short + long term memory
│   ├── short_term.py          # Per-session agent context
│   ├── long_term.py           # ChromaDB vector store
│   ├── rag_retriever.py       # Queries UE5 docs at runtime
│   └── task_history.py        # Logs past tasks + outcomes
│
├── knowledge/                 # RAG source documents
│   ├── ue5_python_api.md
│   ├── blueprint_patterns.md
│   ├── unreal_classes.md
│   └── task_examples.jsonl
│
├── sandbox/                   # Safe code execution
│   ├── code_validator.py      # AST check before running
│   ├── dry_run.py             # Simulate without UE5 writes
│   └── rollback.py            # Undo last N UE5 changes
│
├── config/                    # Typed configuration (Pydantic)
│   ├── settings.py            # All env vars, validated on startup
│   └── constants.py           # Agent roles, model names, limits
│
├── events/                    # Internal async event bus
│   ├── bus.py                 # asyncio.Queue-based pub/sub
│   └── event_types.py         # Typed event dataclasses
│
├── ui/
│   ├── cli.py                 # Terminal interface
│   ├── web_ui.py              # Gradio live dashboard
│   └── agent_monitor.py       # Watch all agents live
│
├── tests/
│   ├── test_tools.py
│   ├── test_agents.py
│   └── test_ue5_mock.py
│
└── logs/
    ├── agent_runs/            # Per-task JSON logs
    └── errors/                # UE5 crash + error logs
```

---

## Development guide

### Running tests

```bash
pytest tests/ -v

# With UE5 mock (no editor needed)
pytest tests/test_ue5_mock.py -v
```

### Adding a new agent

1. Create `agents/your_agent.py` extending `BaseAgent`
2. Add a model assignment in `llm/model_router.py`
3. Add a system prompt in `llm/prompt_templates.py`
4. Register it in `orchestrator/orchestrator.py`

### Adding a new tool

1. Create `tools/your_tool.py` implementing the tool interface
2. Register in `tools/tool_registry.py`
3. Add AST-safe patterns in `sandbox/code_validator.py` if needed

---


## License

MIT — do whatever you want. A star on the repo is appreciated.

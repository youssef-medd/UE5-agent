# AXIOM — AI-Native Unreal Engine 5 Automation Platform

> **Turn plain English into production-ready Unreal Engine 5 code, in seconds.**
> Powered by local LLMs, a multi-agent pipeline, and deep UE5 integration.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue" />
  <img src="https://img.shields.io/badge/Unreal%20Engine-5.3%2B-orange" />
  <img src="https://img.shields.io/badge/LLM-Ollama%20%7C%20Groq-purple" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Early%20Access-red" />
</p>

---

## What is AXIOM?

AXIOM is a **multi-agent AI platform** that acts as a senior Unreal Engine developer living inside your editor. You describe what you want in plain English — it plans, writes, validates, and executes the code directly inside your project, with no manual copy-paste and no cloud lock-in.

```
"Add a gravity gun mechanic with physics-based grab and throw"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                         AXIOM                               │
│  Planner → Coder → Reviewer → Sandbox → Executor           │
│  RAG memory · Event bus · Rollback · Live monitoring        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Working Blueprint + Python code running live in UE5 editor
```

**Everything runs on your machine by default.** Groq cloud is an optional fallback — never required.

---

## Feature Overview

### Core Platform
- **Natural language → UE5 code** — Blueprint, Python, or both
- **Multi-agent pipeline** — specialized agents for planning, coding, reviewing, and execution
- **Local-first** — Ollama backend, no internet required after setup
- **Cloud fallback** — automatic Groq routing when local GPU is saturated
- **Dry-run mode** — preview every change before it touches your project
- **One-click rollback** — undo the last N operations instantly

### Intelligence Layer
- **RAG-powered context** — agents query your UE5 API docs, patterns, and past tasks at runtime
- **Short + long-term memory** — agents remember your project conventions across sessions
- **Role-specific LLMs** — each agent uses the model best suited for its job
- **Adaptive retry** — exponential backoff with jitter on every LLM call
- **Token budget tracking** — hard limits to prevent runaway LLM costs

### Code Safety
- **AST static analysis** — catches dangerous patterns before execution (no `eval`, no `subprocess`, no mass-delete)
- **Risk scoring** — every generated code block gets a low/medium/high risk label
- **Reviewer agent** — independent model validates all code before it runs
- **Rollback stack** — persistent JSON log of every executed operation
- **Sandbox simulation** — full dry-run report with warnings before any live write

### UE5 Integration
- **Remote Control API bridge** — full HTTP integration with UE5 port 30010
- **UE5 Python bridge** — spawn, move, delete actors; modify properties; run scripts
- **Blueprint writer** — creates and compiles Blueprint assets programmatically
- **Asset manager** — imports FBX, textures, materials; queries the asset registry
- **World query** — reads live world state (actor list, transforms, level info)
- **Log watcher** — tails the UE5 output log in real time; surfaces errors immediately

### Developer Experience
- **Interactive CLI** — Rich-powered terminal UI with live status
- **Web dashboard** — Gradio UI for non-technical team members
- **Agent monitor** — live terminal view of every agent's state
- **Task history** — every run saved to JSON for review and replay
- **Event bus** — async pub/sub between all agents and tools
- **Priority task queue** — high-priority requests skip the queue

### Infrastructure
- **Redis state backend** — shared agent state, gracefully falls back to file-based storage
- **ChromaDB vector store** — RAG knowledge base, persisted to disk
- **Pydantic settings** — all config validated on startup, zero silent misconfigurations
- **Structured logging** — timestamped, level-filtered, per-agent log streams

---

## Architecture

```
                          User Prompt
                               │
                               ▼
          ┌────────────────────────────────────────┐
          │              Orchestrator               │
          │  task_parser · task_queue · state_mgr   │
          │  Shared Redis / JSON state              │
          └──────┬────────┬────────┬───────┬───────┘
                 │        │        │       │
                 ▼        ▼        ▼       ▼
            Planner    Coder   Reviewer  Executor
             Agent     Agent    Agent    Agent
           Qwen2.5   Qwen2.5  Mistral  (deterministic)
             14B      Coder     7B
                       32B
                 │        │        │       │
                 └────────┴────────┴───────┘
                               │
                    ┌──────────┴──────────┐
                    │      LLM Layer       │
                    │  Ollama · Groq       │
                    │  ModelRouter         │
                    │  PromptTemplates     │
                    │  RetryMixin          │
                    │  TokenBudget         │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼──────────────────────┐
          ▼                    ▼                       ▼
     Tool Layer           Memory Layer           Sandbox Layer
  ─────────────        ─────────────────       ───────────────
  remote_control       short_term.py           code_validator
  ue5_python_bridge    long_term (ChromaDB)    dry_run
  blueprint_writer     rag_retriever           rollback
  asset_manager        task_history
  world_query          
  log_watcher          
  tool_registry        
          │                    │                       │
          └────────────────────┴───────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │      Event Bus       │
                    │  asyncio pub/sub     │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
            CLI UI         Web UI (Gradio)  Agent Monitor
```

---

## Agent Roles

| Agent | Model | Responsibility |
|---|---|---|
| **Planner** | Qwen2.5-14B | Parses intent, decomposes into ordered subtasks |
| **Coder** | Qwen2.5-Coder-32B | Writes UE5 Python / Blueprint code per step |
| **Reviewer** | Mistral-7B | Validates correctness, flags safety risks, assigns risk level |
| **Executor** | Deterministic | Sends approved code to UE5 via Remote Control API |

Each agent is independently routable — swap any model without touching the others.

---

## Roadmap

### v0.1 — Foundation ✅
- [x] Async LLM abstraction (Ollama + Groq backends)
- [x] Retry logic with exponential backoff
- [x] Token budget tracking
- [x] Config layer with Pydantic validation

### v0.2 — Agent Pipeline ✅
- [x] BaseAgent abstract class
- [x] Planner, Coder, Reviewer, Executor agents
- [x] Event bus with typed events
- [x] Orchestrator with full pipeline routing

### v0.3 — UE5 Integration ✅
- [x] Remote Control HTTP client
- [x] UE5 Python bridge (spawn/delete/move)
- [x] Blueprint writer
- [x] Asset manager + world query
- [x] Live log watcher

### v0.4 — Memory & Safety ✅
- [x] AST code validator
- [x] Dry-run simulation
- [x] Rollback manager
- [x] Short-term memory (sliding window)
- [x] ChromaDB long-term memory
- [x] RAG retriever
- [x] Task history logger

### v0.5 — UI & DX ✅
- [x] Rich CLI with live status
- [x] Gradio web dashboard
- [x] Agent monitor (live terminal)
- [x] Priority task queue
- [x] Redis + file-based state manager

### v0.6 — Multi-Project & Sessions 🔜
- [ ] Named project profiles (switch between UE5 projects instantly)
- [ ] Session persistence — resume a task after editor restart
- [ ] Multi-level undo with named snapshots
- [ ] Agent conversation history (follow-up prompts in context)
- [ ] Inline diff view before execution

### v0.7 — Plugin & Editor Integration 🔜
- [ ] Native UE5 editor plugin (Python Script Plugin extension)
- [ ] Right-click → "Ask AXIOM" context menu on any asset
- [ ] AXIOM panel docked inside Unreal Editor
- [ ] Hotkey to open the prompt bar from inside the editor
- [ ] Blueprint node graph visualization in terminal

### v0.8 — Team & Collaboration 🔜
- [ ] Multi-user mode — shared task queue over local network
- [ ] Task assignment by team member role
- [ ] Approval workflow — reviewer must be a human, not just the AI
- [ ] Shared knowledge base across a studio team
- [ ] Slack / Discord bot integration for remote prompts

### v0.9 — Intelligence Upgrades 🔜
- [ ] Fine-tuned Coder model on UE5 Python corpus
- [ ] RAG auto-indexing — watches your `/Source` and re-indexes on save
- [ ] Long-context planning (50k+ token budget for complex features)
- [ ] Code diff learning — agent improves from accepted/rejected reviews
- [ ] Parallel agent execution for independent subtasks
- [ ] Self-healing — executor detects UE5 Python errors and retries with coder

### v1.0 — Production 🔜
- [ ] Desktop app (Electron or Tauri wrapper)
- [ ] One-click install (no manual Ollama setup)
- [ ] Cloud tier (hosted inference, no local GPU required)
- [ ] UE5 Marketplace plugin listing
- [ ] AXIOM Studio — multi-project dashboard with analytics
- [ ] API access for CI/CD pipelines (auto-generate code on Git push)

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Local LLM runtime | Ollama |
| Cloud LLM fallback | Groq API |
| Planning model | Qwen2.5-14B |
| Coding model | Qwen2.5-Coder-32B |
| Review model | Mistral-7B |
| HTTP client | httpx (async) |
| Config validation | Pydantic + pydantic-settings |
| Vector store | ChromaDB |
| State backend | Redis (file fallback) |
| Terminal UI | Rich + Typer |
| Web UI | Gradio |
| Testing | pytest + pytest-asyncio |
| UE5 bridge | Remote Control API (HTTP) |
| UE5 scripting | Python Script Plugin (unreal module) |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Use pyenv or conda |
| Ollama | latest | ollama.com |
| Unreal Engine | 5.3+ | Remote Control Plugin required |
| Redis | 7+ | Optional — file fallback available |
| ChromaDB | via pip | Auto-initialized on first run |
| GPU (optional) | 12GB+ VRAM | Required for Coder-32B locally |

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

### 2. Pull local models

```bash
ollama pull qwen2.5:14b             # Planner (~9 GB)
ollama pull qwen2.5-coder:32b       # Coder (~20 GB)
ollama pull mistral:7b              # Reviewer (~4 GB)
```

> **Low VRAM?** AXIOM automatically falls back to Groq for any model that doesn't fit. Set `GROQ_API_KEY` and it handles the rest.

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` — at minimum, set your UE5 port:

```env
UE5_REMOTE_CONTROL_PORT=30010
```

### 4. Enable UE5 Remote Control

In Unreal Editor:
1. **Edit → Plugins → Remote Control API** → enable
2. **Edit → Project Settings → Plugins → Remote Control** → HTTP Server Port: `30010`
3. Restart the editor

### 5. Build the knowledge base

```bash
python scripts/build_knowledge_base.py
```

Indexes all UE5 Python API docs, Blueprint patterns, and task examples into ChromaDB for RAG retrieval.

---

## Running

```bash
# Interactive CLI (default)
python main.py

# Web dashboard
python main.py --ui web

# Single prompt, headless
python main.py --prompt "add a point light above the player spawn"

# Dry-run (preview only, no UE5 writes)
python main.py --prompt "spawn 10 crates in a grid" --dry-run

# Share web UI publicly
python main.py --ui web --share
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `GROQ_API_KEY` | _(none)_ | Enables Groq cloud fallback |
| `PLANNER_MODEL` | `qwen2.5:14b` | LLM for planning agent |
| `CODER_MODEL` | `qwen2.5-coder:32b` | LLM for coding agent |
| `REVIEWER_MODEL` | `mistral:7b` | LLM for reviewer agent |
| `UE5_REMOTE_CONTROL_HOST` | `localhost` | UE5 editor host |
| `UE5_REMOTE_CONTROL_PORT` | `30010` | UE5 Remote Control port |
| `REDIS_URL` | `redis://localhost:6379/0` | State backend (optional) |
| `DRY_RUN` | `false` | Global dry-run toggle |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Project Structure

```
ue5-agent/
├── main.py                        # Entry point (CLI / web / headless)
├── requirements.txt
├── .env.example
├── pytest.ini
│
├── config/
│   ├── settings.py                # Pydantic env settings, validated on startup
│   └── constants.py               # Agent roles, model names, token limits
│
├── orchestrator/
│   ├── orchestrator.py            # Full pipeline: plan → code → review → execute
│   ├── task_parser.py             # Intent + flag extraction from raw prompt
│   ├── task_queue.py              # Async priority queue
│   └── state_manager.py           # Redis / JSON shared agent state
│
├── agents/
│   ├── base_agent.py              # Abstract base: LLM routing, event emission
│   ├── planner_agent.py           # Qwen2.5-14B — task decomposition
│   ├── coder_agent.py             # Qwen2.5-Coder-32B — UE5 Python generation
│   ├── reviewer_agent.py          # Mistral-7B — safety + correctness review
│   └── executor_agent.py          # Sends validated code to UE5
│
├── llm/
│   ├── base_llm.py                # Abstract LLM interface, retry, token budget
│   ├── ollama_backend.py          # Local Ollama async backend
│   ├── groq_backend.py            # Groq cloud fallback with rate limit handling
│   ├── model_router.py            # Role → model → backend routing
│   └── prompt_templates.py        # System prompts per agent role
│
├── tools/
│   ├── tool_registry.py           # Register + dispatch agent tools
│   ├── remote_control.py          # UE5 Remote Control HTTP client
│   ├── ue5_python_bridge.py       # Spawn / delete / move actors
│   ├── blueprint_writer.py        # Create + compile Blueprint assets
│   ├── log_watcher.py             # Tail UE5 output log live
│   ├── asset_manager.py           # Import meshes, materials; query registry
│   └── world_query.py             # Read live world state
│
├── memory/
│   ├── short_term.py              # Per-session sliding context window
│   ├── long_term.py               # ChromaDB vector store
│   ├── rag_retriever.py           # Query UE5 docs at runtime
│   └── task_history.py            # Persist + reload task records
│
├── sandbox/
│   ├── code_validator.py          # AST analysis — blocks dangerous patterns
│   ├── dry_run.py                 # Simulate execution without UE5 writes
│   └── rollback.py                # Persistent undo stack
│
├── events/
│   ├── bus.py                     # Async pub/sub event bus
│   └── event_types.py             # Typed event dataclasses
│
├── ui/
│   ├── cli.py                     # Rich interactive terminal UI
│   ├── web_ui.py                  # Gradio web dashboard
│   └── agent_monitor.py           # Live agent state terminal view
│
├── knowledge/
│   ├── ue5_python_api.md          # UE5 Python API reference
│   ├── blueprint_patterns.md      # Common Blueprint code patterns
│   ├── unreal_classes.md          # Key Unreal class reference
│   └── task_examples.jsonl        # Few-shot task examples for RAG
│
├── scripts/
│   └── build_knowledge_base.py    # Index knowledge/ into ChromaDB
│
└── tests/
    ├── test_tools.py              # CodeValidator, DryRunner, TaskParser
    ├── test_agents.py             # Planner, Coder, Reviewer unit tests
    └── test_ue5_mock.py           # Executor with UE5 Remote Control mock
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# UE5 mock only (no editor needed)
pytest tests/test_ue5_mock.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Adding a New Agent

1. Create `agents/your_agent.py` extending `BaseAgent`
2. Add a model assignment in `llm/model_router.py`
3. Add a system prompt in `llm/prompt_templates.py`
4. Wire it into `orchestrator/orchestrator.py`

## Adding a New Tool

1. Create `tools/your_tool.py`
2. Register with `@register("tool_name")` from `tool_registry.py`
3. Add any new AST-safe patterns to `sandbox/code_validator.py` if needed

---

## Contributing

AXIOM is built in the open. PRs, issues, and ideas are welcome.

```bash
# Fork → branch → PR
git checkout -b feat/your-feature
# make changes
git push origin feat/your-feature
# open PR against main
```

**Commit style:** `type: short description`
Types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`

---

## License

MIT — use it, fork it, ship it.
A star on the repo is appreciated.

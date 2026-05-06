# AXIOM — AI-Native Unreal Engine 5 Automation Platform

> **Turn plain English into production-ready Unreal Engine 5 code — in seconds.**
> Multi-agent pipeline · Local LLMs · No cloud lock-in · Full UE5 integration

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Unreal_Engine-5.3+-0e1128?style=for-the-badge&logo=unrealengine&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Groq-Cloud_Fallback-f55036?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Early_Access-ef4444?style=for-the-badge"/>
</p>

---

## What is AXIOM?

AXIOM is a **production-grade multi-agent AI platform** that acts as a senior Unreal Engine developer living inside your editor. You describe what you want in plain English — it plans, writes, validates, and executes code directly inside your UE5 project, with no manual copy-paste and no required internet connection.

```
"Add a gravity gun mechanic with physics-based grab and throw"
          │
          ▼
   ┌─────────────────────────────────────────────────────────┐
   │                       AXIOM                             │
   │   Plan → Code → Review → Validate → Sandbox → Execute  │
   └─────────────────────────────────────────────────────────┘
          │
          ▼
   Working Blueprint + Python code, live in your UE5 editor
```

**Everything runs on your machine by default.** No data leaves your network unless you opt into Groq as a cloud fallback.

---

## System Architecture

```mermaid
flowchart TB
    subgraph ENTRY["🖥️ ENTRY POINTS & INTERFACES"]
        CLI["💻 CLI Terminal - Rich + Typer"]:::eStyle
        WEB["🌐 Web Dashboard - Gradio"]:::eStyle
        HAPI["⚡ Headless API - asyncio"]:::eStyle
        AMON["📡 Agent Monitor - Live view"]:::eStyle
    end

    EBUS["🔄 ASYNC EVENT BUS - asyncio pub/sub - EventKind enum - 15 typed event kinds"]:::busStyle

    subgraph ORCH["🎯 ORCHESTRATOR"]
        TP["task_parser - Intent extraction"]:::orchStyle
        TQ["task_queue - Priority queue"]:::orchStyle
        SM["state_manager - Redis / JSON"]:::orchStyle
    end

    subgraph AGENTS["🤖 AGENT PIPELINE"]
        PA["🧠 PLANNER - Qwen2.5-14B - Task decomposition"]:::agentStyle
        CA["💻 CODER - Qwen2.5-Coder-32B - Code generation"]:::agentStyle
        RA["✅ REVIEWER - Mistral-7B - Safety validation"]:::agentStyle
        EA["⚙️ EXECUTOR - Deterministic - UE5 dispatch"]:::agentStyle
    end

    subgraph LLML["🔮 LLM ABSTRACTION LAYER"]
        MR["ModelRouter"]:::llmStyle
        OB["OllamaBackend"]:::llmStyle
        GB["GroqBackend"]:::llmStyle
        PTE["PromptTemplates"]:::llmStyle
        RMX["RetryMixin"]:::llmStyle
        TBK["TokenBudget"]:::llmStyle
    end

    subgraph TOOLL["🛠️ TOOL LAYER"]
        RCC["RemoteControl - HTTP 30010"]:::toolStyle
        PBR["UE5PythonBridge"]:::toolStyle
        BWR["BlueprintWriter"]:::toolStyle
        AMG["AssetManager"]:::toolStyle
        WQL["WorldQuery"]:::toolStyle
        LWR["LogWatcher"]:::toolStyle
        TRG["ToolRegistry"]:::toolStyle
    end

    subgraph MEML["🧠 MEMORY LAYER"]
        STM["ShortTermMemory - 20-turn window"]:::memStyle
        LTM["LongTermMemory - ChromaDB"]:::memStyle
        RAG["RAGRetriever - Runtime context"]:::memStyle
        TSH["TaskHistory - JSON logs"]:::memStyle
    end

    subgraph SBXL["🛡️ SANDBOX LAYER"]
        CVL["CodeValidator - AST analysis"]:::sbxStyle
        DRL["DryRunner - Simulation"]:::sbxStyle
        RBL["RollbackManager - Undo stack"]:::sbxStyle
    end

    subgraph INFRAL["⚙️ INFRASTRUCTURE & STORAGE"]
        RDB["🔴 Redis - Shared state"]:::infraStyle
        CDB["🟣 ChromaDB - Vector store"]:::infraStyle
        OLS["🦙 Ollama - localhost 11434"]:::infraStyle
        GRQ["☁️ Groq Cloud - api.groq.com"]:::infraStyle
        KBB["📚 Knowledge Base - UE5 Docs"]:::infraStyle
    end

    RCAPI["🔌 UE5 REMOTE CONTROL API - HTTP - Port 30010 - Python Script Plugin"]:::rcStyle

    subgraph UE5ED["🎮 UNREAL ENGINE 5 EDITOR — EXTERNAL PROCESS"]
        BPA["Blueprint Assets"]:::ue5Style
        PYS["Python Scripts"]:::ue5Style
        WST["World State"]:::ue5Style
        ARG["Asset Registry"]:::ue5Style
        OLG["Output Log"]:::ue5Style
    end

    ENTRY --> EBUS
    EBUS --> ORCH
    ORCH --> AGENTS
    AGENTS --> LLML
    LLML --> TOOLL
    LLML --> MEML
    LLML --> SBXL
    TOOLL --> INFRAL
    MEML --> INFRAL
    SBXL --> INFRAL
    INFRAL --> RCAPI
    RCAPI --> UE5ED

    classDef eStyle fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#e0e7ff
    classDef busStyle fill:#1e40af,stroke:#93c5fd,stroke-width:3px,color:#ffffff
    classDef orchStyle fill:#4c1d95,stroke:#a78bfa,stroke-width:2px,color:#ede9fe
    classDef agentStyle fill:#78350f,stroke:#fbbf24,stroke-width:2px,color:#fef3c7
    classDef llmStyle fill:#831843,stroke:#f472b6,stroke-width:2px,color:#fce7f3
    classDef toolStyle fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#d1fae5
    classDef memStyle fill:#713f12,stroke:#fcd34d,stroke-width:2px,color:#fef9c3
    classDef sbxStyle fill:#7f1d1d,stroke:#fca5a5,stroke-width:2px,color:#fee2e2
    classDef infraStyle fill:#111827,stroke:#6b7280,stroke-width:2px,color:#d1d5db
    classDef rcStyle fill:#431407,stroke:#fb923c,stroke-width:2px,color:#ffedd5
    classDef ue5Style fill:#1c1c00,stroke:#facc15,stroke-width:2px,color:#fef9c3
```

---

## Agent Roles

| Agent | Model | Input | Output | Role |
|---|---|---|---|---|
| **Planner** | Qwen2.5-14B | User prompt | JSON step array | Decomposes request into ordered subtasks |
| **Coder** | Qwen2.5-Coder-32B | Step + RAG context | Python code block | Writes UE5 Python or Blueprint scripts |
| **Reviewer** | Mistral-7B | Code block | JSON review object | Validates safety, assigns risk level |
| **Executor** | Deterministic | Code + review | UE5 response | Gates and dispatches to UE5 via HTTP |

---

## Feature Overview

### Core Pipeline
- **Natural language → UE5 code** — Blueprint, Python, or a mix of both
- **4-stage agent pipeline** — Plan → Code → Review → Execute, every time
- **Local-first** — Ollama backend, zero internet required after setup
- **Automatic cloud fallback** — Groq routing when local GPU is overloaded or unavailable
- **Per-step RAG injection** — each agent call is augmented with UE5 API docs at runtime
- **Dry-run mode** — full simulation report before any write reaches UE5
- **One-click rollback** — undo the last N operations from a persistent JSON stack

### Intelligence
- **Multi-model routing** — each agent role maps to the best-fit model
- **Role-specific system prompts** — planner, coder, reviewer, executor each get tuned instructions
- **Short-term memory** — 20-turn sliding context window keeps agents coherent across follow-ups
- **Long-term memory** — ChromaDB vector store persists knowledge across sessions
- **RAG at query time** — UE5 Python API docs, Blueprint patterns, and task examples in ChromaDB
- **Task history** — every run saved as a JSON record, replayable and reviewable
- **Exponential backoff retry** — all LLM calls wrapped in `RetryMixin` with jitter
- **Token budget tracking** — configurable hard limit prevents runaway LLM costs

### Code Safety
- **AST static analysis** — scans every generated script before execution
- **Banned pattern detection** — blocks `eval`, `exec`, `subprocess`, `os.system`, `shutil.rmtree`
- **Banned import detection** — rejects `subprocess`, `socket`, `ctypes`, `pickle`
- **Infinite loop detection** — warns on unconditional `while True` blocks
- **Risk scoring** — reviewer assigns `low / medium / high` to every code block
- **Approval gating** — executor refuses to run anything the reviewer flags `high`
- **Dry-runner** — generates a full simulation report without touching UE5
- **Rollback stack** — persistent undo log, survives editor restarts

### UE5 Integration
- **Remote Control API bridge** — full HTTP integration, UE5 port 30010
- **execute_python** — runs arbitrary Python inside the UE5 Python Script Plugin
- **set_property / get_actor_list** — read and write actor properties directly
- **UE5 Python bridge** — spawn, move, delete, scale actors via unreal module
- **Blueprint writer** — creates Blueprint assets and compiles them programmatically
- **Asset manager** — imports FBX/OBJ/textures, queries the asset registry
- **World query** — reads live world state: actor list, class names, transforms
- **Log watcher** — tails `UnrealEditor.log` in real time, surfaces Python errors immediately
- **Tool registry** — central dispatch layer for all UE5 operations

### Infrastructure
- **Redis state backend** — shared key-value store for inter-agent coordination
- **File-based fallback** — Redis is optional; falls back to JSON automatically
- **ChromaDB on disk** — vector store persists across sessions in `.chromadb/`
- **Knowledge base** — markdown + JSONL documents indexed at startup by `build_knowledge_base.py`
- **Pydantic settings** — all 12 env vars validated on process start
- **Structured logging** — timestamped, level-filtered, per-module log streams
- **Event bus** — asyncio pub/sub for loose coupling between all layers

### Developer Experience
- **Rich CLI** — live spinner, syntax-highlighted code output, color-coded results
- **Gradio web UI** — browser dashboard for non-technical users and QA teams
- **Agent monitor** — live terminal table showing every agent's real-time state
- **Priority task queue** — high-priority prompts skip ahead in a multi-user queue
- **Headless mode** — `--prompt "..."` for scripting, CI pipelines, or editor hooks
- **Unit tests** — pytest suite covering Planner, Coder, Reviewer, Executor, Sandbox, Parser
- **UE5 mock tests** — full executor test suite with Remote Control API mock

---

## Roadmap

### v0.1 — LLM Foundation ✅
- [x] Async LLM abstraction (Ollama + Groq)
- [x] Retry with exponential backoff and jitter
- [x] Token budget tracking
- [x] Typed LLMResponse + GenerationConfig

### v0.2 — Agent Pipeline ✅
- [x] BaseAgent abstract class with LLM routing and event emission
- [x] PlannerAgent — JSON step decomposition
- [x] CoderAgent — code block extraction
- [x] ReviewerAgent — JSON review with risk scoring
- [x] ExecutorAgent — approval gating + UE5 dispatch
- [x] Async event bus with 15 typed event kinds

### v0.3 — UE5 Integration ✅
- [x] Remote Control HTTP client
- [x] UE5 Python bridge (spawn / delete / move)
- [x] Blueprint writer (create + compile)
- [x] Asset manager (import + registry query)
- [x] World query (live actor state)
- [x] Log watcher (real-time UE5 log tail)
- [x] Tool registry

### v0.4 — Memory & Safety ✅
- [x] AST code validator with banned pattern detection
- [x] Dry-run simulation with report
- [x] Persistent rollback manager
- [x] Short-term sliding context window
- [x] ChromaDB long-term memory
- [x] RAG retriever (runtime UE5 doc injection)
- [x] Task history logger

### v0.5 — Orchestration & UI ✅
- [x] TaskParser (intent + flag extraction)
- [x] Priority task queue
- [x] StateManager (Redis + file fallback)
- [x] Full orchestrator pipeline
- [x] Rich CLI + Agent Monitor + Gradio Web UI
- [x] Knowledge base (UE5 API, Blueprint patterns, task examples)
- [x] build_knowledge_base.py indexing script
- [x] Pydantic settings + constants layer
- [x] pytest suite (tools, agents, UE5 mock)

### v0.6 — Sessions & Multi-step Memory 🔜
- [ ] Named project profiles — switch between UE5 projects instantly
- [ ] Session persistence — resume a task after editor restart
- [ ] Multi-level named snapshots (not just undo, but named saves)
- [ ] Agent follow-up — ask "make the light blue" with full prior context
- [ ] Inline diff preview — show what will change before execution
- [ ] Streaming output — print code token-by-token as the model writes it

### v0.7 — Editor Plugin & Native Integration 🔜
- [ ] Native UE5 editor plugin (Python Script Plugin extension)
- [ ] AXIOM panel docked inside Unreal Editor
- [ ] Right-click → "Ask AXIOM" context menu on any asset
- [ ] Hotkey (`Ctrl+Shift+A`) to open prompt bar from inside the editor
- [ ] Blueprint node graph visualization in the terminal
- [ ] Output log parser — detect Python exceptions and auto-retry with Coder

### v0.8 — Team & Collaboration 🔜
- [ ] Multi-user mode — shared task queue over local network
- [ ] Role-based access — which users can approve/reject agent output
- [ ] Human-in-the-loop review — Reviewer agent can escalate to human
- [ ] Shared ChromaDB across a studio team (central server)
- [ ] Slack / Discord bot — submit prompts from your team's chat
- [ ] GitHub Actions integration — generate UE5 code from issue body on PR open

### v0.9 — Self-Improving Intelligence 🔜
- [ ] Fine-tuned Coder model on UE5 Python corpus (QLoRA)
- [ ] RAG auto-indexing — watches `/Source` and re-indexes on save
- [ ] Code diff learning — agent trains on accepted vs rejected reviews
- [ ] Parallel agent execution for independent subtasks
- [ ] Self-healing — executor detects UE5 runtime errors, loops back to Coder
- [ ] Confidence scoring — Planner flags uncertain steps for human confirmation
- [ ] Long-context planning — 50k+ token budget for large feature requests

### v1.0 — Production & Distribution 🔜
- [ ] Desktop app (Tauri wrapper — Windows / macOS)
- [ ] One-click installer (bundles Ollama + AXIOM setup)
- [ ] Cloud tier (hosted inference, no local GPU required)
- [ ] UE5 Marketplace plugin listing
- [ ] AXIOM Studio — multi-project dashboard with cost and quality analytics
- [ ] REST API — integrate AXIOM into your own tools and CI/CD pipelines
- [ ] Webhook triggers — run on Git push, Jira ticket, Slack message

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11+ | Core runtime |
| Local LLM | Ollama | On-device inference |
| Cloud LLM | Groq API | Fallback when local GPU is busy |
| Planning model | Qwen2.5-14B | Natural language → task steps |
| Coding model | Qwen2.5-Coder-32B | UE5 Python / Blueprint generation |
| Review model | Mistral-7B | Safety and correctness validation |
| HTTP client | httpx (async) | Ollama, Groq, UE5 Remote Control |
| Config validation | Pydantic v2 + pydantic-settings | Typed env vars, validated on startup |
| Vector store | ChromaDB | RAG index, persisted to disk |
| State backend | Redis (+ file fallback) | Shared inter-agent state |
| Terminal UI | Rich + Typer | Interactive CLI |
| Web UI | Gradio | Browser-based dashboard |
| Testing | pytest + pytest-asyncio | Unit + mock integration tests |
| UE5 bridge | Remote Control Plugin (HTTP) | Command dispatch to editor |
| UE5 scripting | Python Script Plugin (unreal module) | Actor ops, Blueprint, assets |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | pyenv or conda recommended |
| Ollama | latest | [ollama.com](https://ollama.com) |
| Unreal Engine | 5.3+ | Remote Control Plugin required |
| Redis | 7+ | Optional — file fallback available |
| ChromaDB | via pip | Auto-initialized on first run |
| GPU | 12 GB+ VRAM | For Coder-32B locally; else use Groq |

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

> **Low VRAM?** Set `GROQ_API_KEY` in `.env` — AXIOM falls back to Groq automatically for any model that exceeds local capacity.

### 3. Configure

```bash
cp .env.example .env
```

Minimum required: set your UE5 port if different from the default:

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

Indexes UE5 Python API docs, Blueprint patterns, and task examples into ChromaDB for RAG retrieval at agent runtime.

---

## Running

```bash
# Interactive CLI (default)
python main.py

# Web dashboard
python main.py --ui web

# Single prompt, headless
python main.py --prompt "add a point light above the player spawn"

# Dry-run — preview without writing to UE5
python main.py --prompt "spawn 10 crates in a grid" --dry-run

# Share web UI publicly via Gradio tunnel
python main.py --ui web --share
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `GROQ_API_KEY` | _(none)_ | Enables Groq cloud fallback |
| `PLANNER_MODEL` | `qwen2.5:14b` | Model for planning agent |
| `CODER_MODEL` | `qwen2.5-coder:32b` | Model for coding agent |
| `REVIEWER_MODEL` | `mistral:7b` | Model for reviewer agent |
| `UE5_REMOTE_CONTROL_HOST` | `localhost` | UE5 editor host |
| `UE5_REMOTE_CONTROL_PORT` | `30010` | UE5 Remote Control port |
| `REDIS_URL` | `redis://localhost:6379/0` | State backend (optional) |
| `DRY_RUN` | `false` | Global dry-run toggle |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Project Structure

```
ue5-agent/
├── main.py                        # Entry point — CLI / web / headless modes
├── requirements.txt               # All dependencies
├── .env.example                   # Environment template
├── pytest.ini
│
├── config/
│   ├── settings.py                # Pydantic env vars, validated on startup
│   └── constants.py               # Agent roles, model names, token limits
│
├── orchestrator/
│   ├── orchestrator.py            # Full pipeline: plan → code → review → execute
│   ├── task_parser.py             # Intent + flag extraction from raw prompt
│   ├── task_queue.py              # Async priority queue
│   └── state_manager.py           # Redis / JSON shared agent state
│
├── agents/
│   ├── base_agent.py              # Abstract base: LLM routing, event emission, timing
│   ├── planner_agent.py           # Qwen2.5-14B — JSON step decomposition
│   ├── coder_agent.py             # Qwen2.5-Coder-32B — UE5 Python generation
│   ├── reviewer_agent.py          # Mistral-7B — safety + correctness review
│   └── executor_agent.py          # Approval gating + UE5 dispatch
│
├── llm/
│   ├── base_llm.py                # Abstract LLM interface, RetryMixin, TokenBudget
│   ├── ollama_backend.py          # Local Ollama async streaming backend
│   ├── groq_backend.py            # Groq cloud fallback, rate-limit handling, tracing
│   ├── model_router.py            # Role → model → backend, health-aware routing
│   └── prompt_templates.py        # Typed system prompts per agent role
│
├── tools/
│   ├── tool_registry.py           # Register + dispatch agent tools
│   ├── remote_control.py          # UE5 Remote Control HTTP client
│   ├── ue5_python_bridge.py       # Spawn / delete / move actors
│   ├── blueprint_writer.py        # Create + compile Blueprint assets
│   ├── log_watcher.py             # Async tail of UE5 output log
│   ├── asset_manager.py           # Import + asset registry query
│   └── world_query.py             # Read live world state
│
├── memory/
│   ├── short_term.py              # Sliding context window (20 turns)
│   ├── long_term.py               # ChromaDB vector store
│   ├── rag_retriever.py           # Runtime UE5 doc retrieval
│   └── task_history.py            # Persist + reload task records
│
├── sandbox/
│   ├── code_validator.py          # AST analysis — banned patterns + imports
│   ├── dry_run.py                 # Full simulation without UE5 writes
│   └── rollback.py                # Persistent undo stack (JSON-backed)
│
├── events/
│   ├── bus.py                     # asyncio pub/sub event bus
│   └── event_types.py             # Typed event dataclasses + EventKind enum
│
├── ui/
│   ├── cli.py                     # Rich interactive terminal UI
│   ├── web_ui.py                  # Gradio browser dashboard
│   └── agent_monitor.py           # Live terminal agent state view
│
├── knowledge/
│   ├── ue5_python_api.md          # UE5 Python API reference
│   ├── blueprint_patterns.md      # Common Blueprint code patterns
│   ├── unreal_classes.md          # Key Unreal class + component reference
│   └── task_examples.jsonl        # Few-shot task examples for RAG
│
├── scripts/
│   └── build_knowledge_base.py    # Index knowledge/ into ChromaDB
│
└── tests/
    ├── test_tools.py              # CodeValidator, DryRunner, TaskParser
    ├── test_agents.py             # Planner, Coder, Reviewer unit tests
    └── test_ue5_mock.py           # Executor with Remote Control mock
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# UE5 mock only (no editor needed)
pytest tests/test_ue5_mock.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Extending AXIOM

### Adding an agent

1. Create `agents/your_agent.py` extending `BaseAgent`
2. Set `role = "your_role"` on the class
3. Add a model mapping in `llm/model_router.py`
4. Add a system prompt in `llm/prompt_templates.py`
5. Wire it into `orchestrator/orchestrator.py`

### Adding a tool

1. Create `tools/your_tool.py`
2. Decorate with `@register("tool_name")` from `tool_registry.py`
3. Add new AST-safe patterns to `sandbox/code_validator.py` if it introduces new UE5 calls

---

## Contributing

```bash
git checkout -b feat/your-feature
# make changes, commit each addition separately
git push origin feat/your-feature
# open PR against main
```

**Commit types:** `feat` · `fix` · `refactor` · `test` · `chore` · `docs`

---

## License

MIT — use it, fork it, ship it. A star is appreciated.

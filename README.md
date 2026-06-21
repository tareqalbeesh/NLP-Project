# Forensic Linguist Agent

A self-hosted agentic workflow for **forensic authorship analysis**. The agent
characterises *idiolectal style* — how a text is written rather than what it
is about — using a coordinator LLM that calls reusable stylometric tools and
reasons over the results.

Built on top of the [n8n self-hosted AI starter kit](https://github.com/n8n-io/self-hosted-ai-starter-kit)
(Docker Compose: n8n + Postgres + Qdrant + Ollama) with an added
**FastAPI sidecar** (`nltk-tools`) exposing NLTK / spaCy / scikit-learn
analyses as HTTP endpoints.

---

## What it does

Three task types, each implemented as a separate n8n workflow:

1. **Open-ended stylistic analysis** — agent picks tools, computes features, narrates findings.
2. **Authorship attribution** — *who wrote this?* Deterministic protocol: pre-compute features for unknown + every indexed reference sample, LLM does only the numeric comparison.
3. **Verification baseline (no LLM)** — pure stylometric distance for sanity-checking the agent.

Plus a **corpus-builder** workflow for indexing known-author samples into Qdrant.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Docker network: "demo"                          │
│                                                                          │
│   ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌────────────────┐     │
│   │  n8n    │    │ postgres  │    │  qdrant  │    │    ollama      │     │
│   │ :5678   │◄──►│   :5432   │    │  :6333   │    │    :11434      │     │
│   └────┬────┘    └───────────┘    └────┬─────┘    └────────────────┘     │
│        │              n8n's DB         │ vector store     LLM (local)    │
│        │                               │                                 │
│        │      HTTP tool calls          │                                 │
│        ▼                               ▼                                 │
│   ┌────────────────────────────────────────────────────┐                 │
│   │           nltk-tools (FastAPI, :8000)              │                 │
│   │                                                    │                 │
│   │  Stylometric feature endpoints:                    │                 │
│   │    /char_ngrams, /vocab_richness,                  │                 │
│   │    /function_word_freqs, /pos_distribution,        │                 │
│   │    /sentence_length_stats, /punctuation_profile    │                 │
│   │                                                    │                 │
│   │  Authorship endpoints (backed by qdrant):          │                 │
│   │    /index_author, /list_samples, /list_authors,    │                 │
│   │    /get_sample_text, /compare_two_texts,           │                 │
│   │    /attribute, /extract_all_features,              │                 │
│   │    /extract_corpus_features                        │                 │
│   └────────────────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────────────┘
```

The LLM is reached via an **OpenAI-compatible endpoint** — Webis-hosted (recommended,
`https://chat.web.webis.de/openai/`) or your local Ollama. Tool-capable
models only: `qwen3-30b-a3b` (recommended), `llama3.1:8b`, `mistral-nemo`.
The model `llama3-8b` does **not** support tools and won't work for the agentic workflows.

---

## Repository layout

```
.
├── docker-compose.yml                  # the whole stack
├── .env.example                        # copy to .env
│
├── nltk-tools/                         # FastAPI sidecar
│   ├── app.py                          # ~600 lines, all endpoints
│   ├── Dockerfile
│   └── requirements.txt
│
├── n8n/forensic-workflows/             # importable n8n workflows
│   ├── main_workflow_code_tools.json   # primary agentic workflow (10 tools)
│   ├── attribute_deterministic.json    # authorship attribution
│   ├── baseline_no_llm.json            # verification, no LLM
│   ├── index_author.json               # corpus builder
│   ├── main_workflow_updated.json      # variant using sub-workflows-as-tools
│   └── tool_char_ngrams.json           # example sub-workflow for the variant
│
├── shared/                             # mounted into n8n as /data/shared
│   ├── sample_austen.txt               # Pride and Prejudice opening
│   └── sample_fitzgerald.txt           # The Great Gatsby opening
│
├── functions/                          # standalone Python reference implementation
│   ├── ttr.py, rttr.py                 #   type-token ratio variants
│   ├── yules_k.py, hapax_ratio.py      #   lexical-richness metrics
│   ├── tokenize_char_ngrams.py         #   character n-gram extractor
│   ├── verify_authorship.py            #   full authorship-verification pipeline
│   ├── analyze_text.py                 #   spaCy POS / NER demo
│   ├── extract_entity_relations.py     #   entity extraction
│   ├── resolve_references.py           #   coreference / reference resolution
│   └── report.py                       #   pretty-print a stylometric profile
│
├── readme/                             # extended setup docs
│   ├── DEVCONTAINER.md                 #   VS Code dev container setup
│   ├── OLLAMA_BACKEND.md               #   local Ollama backend setup
│   └── WEBIS_LLM_BACKEND.md            #   Webis LLM backend setup (Continue extension)
│
└── n8n/demo-data/                      # auto-imported on first boot
    ├── credentials/                    # local Ollama, local Qdrant
    └── workflows/                      # n8n's original demo
```

### Two implementations side by side

The repo holds **two parallel approaches** to stylometric analysis:

| Implementation | Where it lives | How to use it |
|---|---|---|
| **HTTP service** (the docker-compose stack) | `nltk-tools/app.py` | Called by n8n workflows via `http://nltk-tools:8000`. Used for the agentic / deterministic / baseline workflows. |
| **Standalone Python modules** | `functions/*.py` | Plain Python — `from functions.verify_authorship import verify_authorship`. Useful in notebooks, scripts, or as a reference for the FastAPI endpoints' logic. |

They overlap on metrics (TTR, Yule's K, hapax ratio, char n-grams) but differ in how they're wired. Pick the one that fits your task: the HTTP service for the LLM-driven agent, the standalone modules for direct scripting / evaluation harnesses.

---

## Prerequisites

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux), running.
- **~10 GB free disk** for images.
- **First build takes ~5 min** (spaCy `en_core_web_sm` + NLTK corpora download into the `nltk-tools` image).
- **Optional GPU** — Nvidia (with [Nvidia Container Toolkit](https://github.com/ollama/ollama/blob/main/docs/docker.md)) or AMD on Linux. CPU works but Llama inference is slow.
- **A Webis API key** for the LLM backend. Each team member gets their own from `klara.gutekunst@uni-kassel.de`. Detailed Webis setup is in [readme/WEBIS_LLM_BACKEND.md](readme/WEBIS_LLM_BACKEND.md); for fully-offline use see [readme/OLLAMA_BACKEND.md](readme/OLLAMA_BACKEND.md).
- **Optional Dev Container** — if you'd rather develop inside a containerized VS Code, see [readme/DEVCONTAINER.md](readme/DEVCONTAINER.md).

---

## Quick start

```bash
git clone https://github.com/tareqalbeesh/NLP-Project.git
cd NLP-Project
```

Configure environment (defaults work for local dev):

```bash
# Windows PowerShell
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

> `.env` is gitignored. Don't commit edits.

Start the stack — pick the profile that matches your hardware:

```bash
# Nvidia GPU
docker compose --profile gpu-nvidia up -d --build

# AMD GPU (Linux only)
docker compose --profile gpu-amd up -d --build

# CPU / Mac
docker compose --profile cpu up -d --build
```

First run downloads images and builds `nltk-tools` (~5 min). Subsequent
runs are fast — drop `--build` unless you've edited `nltk-tools/`.

Verify it's up:

```bash
docker compose ps
```

Five services should be `Up`: `n8n`, `nltk-tools`, `postgres`, `qdrant`, `ollama-gpu`/`-cpu`/`-gpu-amd`. Two helper containers `n8n-import` and `ollama-pull-llama-*` exit cleanly with code 0.

Smoke-test the Python service:

```bash
curl -X POST http://localhost:8000/vocab_richness \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"hello world hello\"}"
```

Expected: JSON with `N`, `V`, `TTR`, `Yule_K`, etc.

---

## First-time n8n setup

1. Open <http://localhost:5678>.
2. Create the owner account (any email / password — local only).
3. **Add the Webis LLM credential**:
   - Credentials → New → **OpenAI**
   - **API Key**: your Webis key
   - **Base URL**: `https://chat.web.webis.de/openai/`
   - Save and Test — should report success.
4. **Import the workflows** — Workflows → "Import from File", one at a time, from `n8n/forensic-workflows/`:

| File | Purpose |
|---|---|
| `main_workflow_code_tools.json` | Open-ended agentic stylometric analysis (10 tools) |
| `attribute_deterministic.json` | Authorship attribution — deterministic protocol |
| `baseline_no_llm.json` | Verification baseline (no LLM) |
| `index_author.json` | Corpus builder — index known-author samples |

5. After importing each workflow, **open the OpenAI Model node** and confirm the model dropdown is `qwen3-30b-a3b` (or another tool-capable Webis model). If the dropdown is red/empty, pick it manually.

---

## Build a reference corpus

The attribution workflow needs known-author samples in Qdrant first.

Open **Index Author Sample** → Execute → submit for each known author:

| Author label | Sample file |
|---|---|
| `austen`      | `shared/sample_austen.txt` |
| `fitzgerald`  | `shared/sample_fitzgerald.txt` |

Each submission returns `status: indexed` plus a corpus snapshot. The
corpus persists in the `qdrant_storage` Docker volume across restarts.

Add as many authors / samples as you like — the attribution workflow
automatically includes them all.

---

## Testing the system

### 1. Agentic stylistic analysis

Open **`Forensic linguist agent`** (the main workflow) → Execute → upload
`shared/sample_austen.txt` with instructions like:

> *Run vocab_richness, function_word_freqs, and pos_distribution on this document. Cite the exact numeric values from each tool's output.*

Watch the canvas — tool nodes should highlight as they fire. The output
should cite real numbers like `TTR: 0.5469`, `N: 309`. **Numbers should match
what `curl` returns from the same endpoint.**

In a second terminal tail the Python service to see calls land:
```bash
docker compose logs -f nltk-tools
```

### 2. Authorship attribution (deterministic)

Open **`Forensic Attribution (deterministic protocol)`** → Execute →
upload `shared/sample_austen.txt`. No instructions field — this workflow
is single-purpose.

Expected: a feature-by-feature comparison table plus a verdict naming
`austen` with high confidence. Exactly **two** lines in
`docker compose logs nltk-tools`: one POST `/extract_all_features` (for the
unknown), one GET `/extract_corpus_features` (for the corpus).

### 3. Verification baseline (no LLM)

Open **`Forensic Linguist Baseline (no LLM)`** → Execute → upload
`shared/sample_austen.txt` as Text A and `shared/sample_fitzgerald.txt`
as Text B.

Output: per-feature divergence + verdict directly from the
`/compare_two_texts` endpoint. No agent. Use this to cross-check what the
LLM agent says.

---

## Why three workflows

The course rubric requires (1) a working agent system, (2) transparency and
reproducibility, and (3) comparison against a non-agentic baseline. Each
workflow is the cleanest implementation of one of those:

- **Main agentic workflow** — demonstrates the agentic pattern (tool
  selection, multi-step reasoning, dynamic behaviour).
- **Deterministic attribution** — guarantees the protocol executes
  end-to-end with inspectable intermediate results, regardless of how
  good the LLM is at planning. Used when correctness matters more than
  reasoning flexibility.
- **No-LLM baseline** — what stylometry can do without any agent. The
  agent's value-add is the difference between these two.

---

## Common issues

| Symptom | Fix |
|---|---|
| `no configuration file provided: not found` | You're in the wrong directory. `cd` to the repo root. |
| n8n shows credential error on workflow open | The credentials are encrypted with `N8N_ENCRYPTION_KEY` and tied to your machine. Re-create the OpenAI credential in n8n's UI. |
| Tool nodes never fire / agent narrates "I'll use a tool..." | Model isn't tool-capable. Switch to `qwen3-30b-a3b` (not `llama3-8b`). |
| `Tool error: fetch is not defined` | Code Tools must use `this.helpers.httpRequest({...})`, not `fetch`. |
| `nltk-tools` keeps restarting | Check `docker compose logs nltk-tools`. Usually Qdrant wasn't ready yet — `docker compose restart nltk-tools` after Qdrant is up. |
| Agent calls a tool but result is wrong numbers | The model truncated/summarised the document. Use the deterministic workflow instead. |
| Webhook ID collision on form submit | You imported two workflows with the same `webhookId`. Delete the old workflow. |

---

## Useful commands

```bash
# Tail FastAPI service logs (every tool call appears here)
docker compose logs -f nltk-tools

# Tail n8n logs (execution errors)
docker compose logs -f n8n

# Restart a single service
docker compose restart n8n
docker compose restart nltk-tools

# Rebuild nltk-tools after editing app.py
docker compose --profile gpu-nvidia up -d --build nltk-tools

# Stop everything (data persists in volumes)
docker compose --profile gpu-nvidia down

# Stop AND delete all data (Qdrant corpus, downloaded models, n8n workflows)
docker compose --profile gpu-nvidia down -v
```

---

## Tech stack

- **Docker Compose** orchestration
- **n8n** — agentic workflow runtime
- **FastAPI** — Python tool backend (NLTK, spaCy, scikit-learn)
- **Qdrant** — vector store for author fingerprints (cosine similarity)
- **Postgres** — n8n's metadata store
- **Ollama** — optional local LLM (Llama 3.2 by default; tool-capable)
- **Webis LLM endpoint** — OpenAI-compatible, recommended for serious tool calling

---

## License

Apache 2.0 — see [LICENSE](LICENSE). This project builds on the
[n8n self-hosted AI starter kit](https://github.com/n8n-io/self-hosted-ai-starter-kit),
also Apache 2.0.

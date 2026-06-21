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
┌────────────────────────────────────────────────────────────────────────────┐
│                          Docker network: "demo"                            │
│                                                                            │
│   ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────┐    │
│   │  n8n    │  │ postgres │  │  redis  │  │ qdrant  │  │   ollama     │    │
│   │ :5678   │  │  :5432   │  │  :6379  │  │  :6333  │  │   :11434     │    │
│   └────┬────┘  └──────────┘  └────┬────┘  └────┬────┘  └──────────────┘    │
│        │      n8n's metadata DB   │ corpus DB  │ vectors    LLM (local)    │
│        │                          │            │                           │
│        │                          ▼            ▼                           │
│        │                  ┌──────────────────────┐                         │
│        │                  │   redis-insight      │  Redis GUI (browser)    │
│        │                  │       :5540          │  http://localhost:5540  │
│        │                  └──────────────────────┘                         │
│        │      HTTP tool calls                                              │
│        ▼                                                                   │
│   ┌──────────────────────────────────────────────────────┐                 │
│   │           nltk-tools (FastAPI, :8000)                │                 │
│   │                                                      │                 │
│   │  Stylometric feature endpoints:                      │                 │
│   │    /char_ngrams, /vocab_richness,                    │                 │
│   │    /function_word_freqs, /pos_distribution,          │                 │
│   │    /sentence_length_stats, /punctuation_profile      │                 │
│   │                                                      │                 │
│   │  Authorship endpoints (backed by qdrant + redis):    │                 │
│   │    /index_author, /list_samples, /list_authors,      │                 │
│   │    /get_sample_text, /compare_two_texts,             │                 │
│   │    /attribute, /extract_all_features,                │                 │
│   │    /extract_corpus_features                          │                 │
│   └──────────────────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────┘
```

### Data layer

The system uses three storage layers, each for a different concern:

| Service | Role | Port | What it holds |
|---|---|---|---|
| **Redis** | Primary corpus store | `6379` | Full text content of every indexed author sample, keyed by `dataset:fandom:pair:<id>`. Loaded from the **PAN20 authorship verification** dataset via the [data-loading/](data-loading/) scripts. |
| **Qdrant** | Vector store | `6333` | Stylometric fingerprint vectors (198-dim function-word frequencies) for nearest-author retrieval. |
| **Postgres** | n8n metadata | (internal) | n8n's own workflows, executions, credentials. Not used directly by the analysis. |
| **RedisInsight** | Redis GUI | `5540` | Browser-based inspector for Redis — see <http://localhost:5540> to browse the indexed corpus visually. |

The LLM is reached via an **OpenAI-compatible endpoint** — Webis-hosted (recommended,
`https://chat.web.webis.de/openai/`) or your local Ollama. Tool-capable
models only: `qwen3-30b-a3b` (recommended), `llama3.1:8b`, `mistral-nemo`.
The model `llama3-8b` does **not** support tools and won't work for the agentic workflows.

---

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

Seven services should be `Up`: `n8n`, `nltk-tools`, `postgres`, `redis`,
`redis-insight`, `qdrant`, `ollama-gpu`/`-cpu`/`-gpu-amd`. Two helper
containers `n8n-import` and `ollama-pull-llama-*` exit cleanly with code 0.

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
| ★ `forensic_linguist_agent.json` | **Primary** — chat-based agent, all 6 tools inline JS, Ollama + Redis memory. Start here. |
| `main_workflow_code_tools.json` | Alternative agent (form-based, 10 tools, HTTP-calls into nltk-tools) |
| `attribute_deterministic.json` | Authorship attribution — deterministic protocol |
| `baseline_no_llm.json` | Verification baseline (no LLM) |
| `index_author.json` | Corpus builder — index known-author samples |

5. **After importing the primary workflow `forensic_linguist_agent.json`, re-attach the credentials** (n8n strips them on export so they show as red/missing):
   - **Ollama Chat Model node** → click it → under Credential, pick or create an Ollama credential pointing at `http://ollama:11434`. The workflow's default model is `minimax-m3` — if you don't have it pulled, change the dropdown to any tool-capable Ollama model you do have (`llama3.2`, `qwen2.5`, etc.) or run `docker compose exec ollama-gpu ollama pull minimax-m3`.
   - **Redis Chat Memory node** → pick or create a Redis credential pointing at host `redis`, port `6379`, db `0` (the same Redis service the corpus uses; n8n keys go in a separate namespace).
6. For the **other workflows** (`main_workflow_code_tools`, `attribute_deterministic`), open the OpenAI Model node and confirm the model is `qwen3-30b-a3b` (or another tool-capable Webis model).

---

## Build a reference corpus

You have two options depending on how large a corpus you want.

### Option A — small handcrafted corpus (via n8n)

For a few known samples (good for testing): open **Index Author Sample**
in n8n → Execute → submit for each known author:

| Author label | Sample file |
|---|---|
| `austen`      | `shared/sample_austen.txt` |
| `fitzgerald`  | `shared/sample_fitzgerald.txt` |

Each submission returns `status: indexed` plus a corpus snapshot.

### Option B — bulk-load the PAN20 corpus into Redis

For a real evaluation corpus, use the [data-loading/](data-loading/)
Python scripts. They read the **PAN20 authorship verification**
dataset (`pan20-authorship-verification-test.jsonl`) and load each
text pair into Redis under the key pattern `dataset:fandom:pair:<id>`,
storing `text_1`, `text_2`, `fandom_1`, `fandom_2`, and `pair_id`.

Download the PAN20 dataset and place the JSONL inside `dataset/` (which
is gitignored). Then from the repo root, in a Python env that has the
`redis` package installed:

```bash
python data-loading/data-loading.py     # bulk-load PAN20 pairs into Redis
python data-loading/reading-authors.py  # scan Redis and enumerate unique fandoms
```

After loading, browse the indexed corpus visually at
**<http://localhost:5540>** (RedisInsight). The corpus persists in
`./redis-data/` (mounted as a bind volume) across container restarts.

Either way, the attribution workflow automatically includes everything
in the corpus — no n8n changes needed.

---

## Testing the system

### 1. Primary workflow — chat-based agent ★

Open **`Forensic Linguist Agent`** (the imported primary workflow) → click
**"Open Chat"** (bottom of the canvas). Paste two sample passages directly
into the chat — no file upload needed:

> *Were these two passages written by the same author?*
> *Text A: "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families..."*
> *Text B: "In my younger and more vulnerable years my father gave me some advice that I have been turning over in my mind ever since. Whenever you feel like criticizing anyone, he told me, just remember that all the people in this world haven't had the advantages that you've had..."*

The agent should call `verify_authorship`, `compare_stylistic_profiles`,
and `vocabulary_richness` automatically, then return a structured
verdict: **Same/Different Author + Confidence Score + Evidence**. All
math runs inline as JavaScript — no `nltk-tools` service required for
this workflow.

Watch the canvas — tool nodes should highlight as they fire. Conversation
history is stored in Redis so follow-ups like *"Now compare against this
third sample…"* work.

### 2. Alternative agentic workflow (HTTP-based)

If you want to test the FastAPI-backed agent instead, open
**`Forensic linguist agent`** (the form-based workflow) → Execute → upload
`shared/sample_austen.txt` with instructions like:

> *Run vocab_richness, function_word_freqs, and pos_distribution on this document. Cite the exact numeric values from each tool's output.*

Tail the Python service to see calls land:
```bash
docker compose logs -f nltk-tools
```

The output should cite real numbers like `TTR: 0.5469`, `N: 309` — these
should match what `curl http://localhost:8000/vocab_richness` returns.

### 3. Authorship attribution (deterministic)

Open **`Forensic Attribution (deterministic protocol)`** → Execute →
upload `shared/sample_austen.txt`. No instructions field — this workflow
is single-purpose.

Expected: a feature-by-feature comparison table plus a verdict naming
`austen` with high confidence. Exactly **two** lines in
`docker compose logs nltk-tools`: one POST `/extract_all_features` (for the
unknown), one GET `/extract_corpus_features` (for the corpus).

### 4. Verification baseline (no LLM)

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
- **Redis** — primary corpus store (full text of indexed author samples)
- **RedisInsight** — browser-based Redis GUI (`localhost:5540`)
- **Qdrant** — vector store for author fingerprints (cosine similarity)
- **Postgres** — n8n's metadata store
- **Ollama** — optional local LLM (Llama 3.2 by default; tool-capable)
- **Webis LLM endpoint** — OpenAI-compatible, recommended for serious tool calling

---

## License

Apache 2.0 — see [LICENSE](LICENSE). This project builds on the
[n8n self-hosted AI starter kit](https://github.com/n8n-io/self-hosted-ai-starter-kit),
also Apache 2.0.

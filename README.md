# SHL Conversational Assessment Recommender

A stateless FastAPI service that turns a vague hiring need into a grounded shortlist of
SHL Individual Test Solutions through multi-turn dialogue. See `CLAUDE.md` for the full
design contract.

## Stack

- **API:** FastAPI + Uvicorn
- **LLM:** Gemini (`gemini-2.5-flash-lite`) via the OpenAI client against Google's
  OpenAI-compatible endpoint
- **Retrieval:** FAISS (CPU) over precomputed catalog embeddings
- **Package manager:** `uv`

## Setup

```bash
uv sync                 # create .venv and install deps
cp .env.example .env    # then fill in GEMINI_API_KEY
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Run the LangGraph dev server** (primary way to run the agent):
```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

**Install dependencies** (uses uv):
```bash
pip install -e ".[dev]"
```

**Lint**:
```bash
ruff check src/
ruff format src/
```

**Evaluation**:
```bash
# Create the LangSmith dataset first
python eval/create_dataset.py

# Run evaluation against a running agent server
python eval/run_eval.py --experiment-prefix "My prefix" --agent-url http://localhost:2024
```

## Environment Setup

Copy `.env.example` to `.env` and populate:
- `ANTHROPIC_API_KEY` — required for default LLM provider
- `TAVILY_API_KEY` — required for default search provider
- `BRAVE_SEARCH_API_KEY` — required only if `search_provider=brave`
- `SERPER_API_KEY` — required only if `search_provider=serper`

## Architecture

The agent is a **LangGraph state machine** defined in `src/agent/graph.py` and exposed as `graph` (referenced by `langgraph.json`). It follows a research → extract → reflect loop:

```
START → generate_queries → research_person → gather_notes_extract_schema → reflection → (END or research_person)
```

**Node responsibilities:**
- `generate_queries`: Uses LLM to produce targeted web search queries from the person's identity and extraction schema
- `research_person`: Executes queries concurrently (Tavily/Brave/Serper), optionally deep-scrapes URLs via Jina Reader, then has the LLM produce structured research notes
- `gather_notes_extract_schema`: Consolidates all notes and uses `model.with_structured_output(schema)` to extract the final dict
- `reflection`: Evaluates completeness; if unsatisfactory and under `max_reflection_steps`, routes back to `research_person` with new queries

**Key files:**
- `src/agent/state.py` — `InputState`, `OverallState`, `OutputState` dataclasses + `Person` Pydantic model + `DEFAULT_EXTRACTION_SCHEMA`
- `src/agent/configuration.py` — `Configuration` dataclass; fields can be overridden via environment variables (uppercased) or `RunnableConfig`
- `src/agent/prompts.py` — All four LLM prompts (`QUERY_WRITER_PROMPT`, `INFO_PROMPT`, `EXTRACTION_PROMPT`, `REFLECTION_PROMPT`)
- `src/agent/utils.py` — `deduplicate_and_format_sources` and `format_all_notes` helpers

**Configuration** (all overridable at runtime via `configurable` dict or env vars):
- `max_search_queries` (default: 3), `max_search_results` (default: 3), `max_reflection_steps` (default: 0)
- `search_provider`: `"tavily"` | `"brave"` | `"serper"`
- `enable_deep_scrape`: when `True`, fetches top 2 URLs through `https://r.jina.ai/`
- `llm_provider`: `"anthropic"` | `"ollama"`; if `"ollama"`, uses `ollama_model` (default: `"llama3"`)

**Schema requirements:** JSON schemas passed as `extraction_schema` must include top-level `title` and `description` fields, and should avoid deep nesting (LLM extraction degrades with nested objects).

**`completed_notes`** in `OverallState` uses `Annotated[list, operator.add]` — notes accumulate across reflection iterations rather than being overwritten.

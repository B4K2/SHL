import json

from openai import AsyncOpenAI

from app.catalog import CatalogRecord
from app.config import Settings
from app.embeddings import embed_query
from app.vector_store import Predicate, VectorStore

SEARCH_CATALOG_TOOL = {
    "type": "function",
    "function": {
        "name": "search_catalog",
        "description": (
            "Semantic search over SHL Individual Test Solutions. Returns up to 10 candidate "
            "assessments with their catalog id. Optional filters narrow results to stated "
            "constraints only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the role, skills, and context.",
                },
                "test_type": {
                    "type": "string",
                    "description": "Single SHL test-type letter to require: A B C D E K P S.",
                },
                "max_duration_minutes": {
                    "type": "integer",
                    "description": "Only assessments at or under this many minutes. Items with "
                    "unknown duration are kept.",
                },
                "job_level": {
                    "type": "string",
                    "description": "Required job level, e.g. 'Mid-Professional', 'Manager', 'Director'.",
                },
                "language": {
                    "type": "string",
                    "description": "Required language, e.g. 'French', 'English (USA)'. The languages "
                    "list in results may be truncated ('+N more') — to check whether an assessment "
                    "supports a language, search WITH this filter instead of inspecting the list.",
                },
            },
            "required": ["query"],
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Return the final response to the user for this turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": "Natural-language text shown to the user. Refer to assessments "
                    "by name only — NEVER include catalog ids or URLs in this text.",
                },
                "recommendation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Catalog ids of recommended items, in priority order. Empty while "
                    "gathering context or refusing.",
                },
                "end_of_conversation": {"type": "boolean"},
            },
            "required": ["reply", "recommendation_ids", "end_of_conversation"],
        },
    },
}

TOOLS = [SEARCH_CATALOG_TOOL, SUBMIT_TOOL]


def _build_predicate(
    test_type: str | None,
    max_duration_minutes: int | None,
    job_level: str | None,
    language: str | None,
) -> Predicate | None:
    if not any([test_type, max_duration_minutes, job_level, language]):
        return None

    def predicate(record: CatalogRecord) -> bool:
        if test_type and record.test_type != test_type.strip().upper():
            return False
        if max_duration_minutes is not None:
            minutes = record.duration_minutes()
            if minutes is not None and minutes > max_duration_minutes:
                return False
        if job_level and not any(job_level.lower() in jl.lower() for jl in record.job_levels):
            return False
        if language and not any(language.lower() in lang.lower() for lang in record.languages):
            return False
        return True

    return predicate


def _candidate(record: CatalogRecord) -> dict:
    languages = record.languages
    if len(languages) > 8:
        languages = languages[:6] + [f"+{len(record.languages) - 6} more"]
    return {
        "id": record.id,
        "name": record.name,
        "test_type": record.test_type,
        "duration": record.duration or "unspecified",
        "job_levels": record.job_levels,
        "languages": languages,
        "description": record.description[:240],
    }


async def run_search_catalog(
    client: AsyncOpenAI,
    settings: Settings,
    store: VectorStore,
    arguments: dict,
) -> str:
    query_vector = await embed_query(
        client,
        settings.embedding_model,
        settings.embedding_dimensions,
        arguments["query"],
    )
    predicate = _build_predicate(
        arguments.get("test_type"),
        arguments.get("max_duration_minutes"),
        arguments.get("job_level"),
        arguments.get("language"),
    )
    results = store.search(query_vector, k=settings.search_k, predicate=predicate)
    records = [record for record, _ in results]
    seen_ids = {record.id for record in records}
    for exact in store.match_name(arguments["query"]):
        if exact.id in seen_ids:
            continue
        if predicate and not predicate(exact):
            continue
        records.insert(0, exact)
        seen_ids.add(exact.id)

    payload: dict = {"results": [_candidate(record) for record in records[: settings.search_k]]}
    if not records and predicate is not None:
        payload["note"] = (
            "The filters eliminated every match. Retry without filters (or with fewer), then tell "
            "the user which stated constraint the catalog cannot satisfy."
        )
    return json.dumps(payload)

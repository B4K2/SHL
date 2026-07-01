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
                    "description": "Only assessments at or under this many minutes.",
                },
                "job_level": {
                    "type": "string",
                    "description": "Required job level, e.g. 'Mid-Professional', 'Manager', 'Director'.",
                },
                "language": {
                    "type": "string",
                    "description": "Required language, e.g. 'French', 'English (USA)'.",
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
                "reply": {"type": "string"},
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
            if minutes is None or minutes > max_duration_minutes:
                return False
        if job_level and not any(job_level.lower() in jl.lower() for jl in record.job_levels):
            return False
        if language and not any(language.lower() in lang.lower() for lang in record.languages):
            return False
        return True

    return predicate


def _candidate(record: CatalogRecord) -> dict:
    return {
        "id": record.id,
        "name": record.name,
        "test_type": record.test_type,
        "duration": record.duration or "unspecified",
        "job_levels": record.job_levels,
        "languages": record.languages[:6],
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
    results = store.search(query_vector, k=settings.max_recommendations, predicate=predicate)
    return json.dumps({"results": [_candidate(record) for record, _ in results]})

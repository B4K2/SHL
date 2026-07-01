import asyncio
import json
import re

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from app.config import Settings
from app.prompts import SYSTEM_PROMPT
from app.schemas import ChatResponse, Message, Recommendation
from app.tools import TOOLS, run_search_catalog
from app.vector_store import VectorStore

MAX_TOOL_ROUNDS = 6
MAX_GENERATION_RETRIES = 2
TRANSIENT_ERRORS = (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError)

_REFUSAL_FALLBACK = ChatResponse(
    reply="I can only help with selecting SHL assessments. Could you tell me about the role you're hiring for?",
    recommendations=[],
    end_of_conversation=False,
)


class Agent:
    def __init__(self, client: AsyncOpenAI, settings: Settings, store: VectorStore) -> None:
        self._client = client
        self._settings = settings
        self._store = store

    # Added limit so that the model doesnt respond after 8 conv
    async def respond(self, messages: list[Message]) -> ChatResponse:
        for _ in range(MAX_GENERATION_RETRIES):
            response = await self._run_turn(messages)
            if response is not None:
                return response
        return _REFUSAL_FALLBACK

    async def _run_turn(self, messages: list[Message]) -> ChatResponse | None:
        conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        conversation.extend({"role": m.role, "content": m.content} for m in messages)

        for round_index in range(MAX_TOOL_ROUNDS):
            force_submit = round_index == MAX_TOOL_ROUNDS - 1
            completion = await self._complete(conversation, force_submit)
            message = completion.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                return None

            conversation.append(_assistant_dict(message))

            submitted = self._find_submit(tool_calls)
            if submitted is not None:
                return submitted

            for call in tool_calls:
                result = await self._dispatch_tool(call)
                conversation.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        return None

    async def _complete(self, conversation: list[dict], force_submit: bool):
        tool_choice = (
            {"type": "function", "function": {"name": "submit"}} if force_submit else "required"
        )
        backoff = 2.0
        last_error: Exception | None = None
        for _ in range(self._settings.max_completion_retries):
            try:
                return await self._client.chat.completions.create(
                    model=self._settings.chat_model,
                    messages=conversation,
                    tools=TOOLS,
                    tool_choice=tool_choice,
                )
            except RateLimitError as error:
                last_error = error
                await asyncio.sleep(min(_retry_after(error, default=backoff), 30.0))
                backoff = min(backoff * 2, 30.0)
            except TRANSIENT_ERRORS as error:
                last_error = error
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        raise last_error

    def _find_submit(self, tool_calls: list) -> ChatResponse | None:
        for call in tool_calls:
            if call.function.name == "submit":
                arguments = _safe_json(call.function.arguments)
                if arguments is None:
                    return None
                return self._build_response(arguments)
        return None

    async def _dispatch_tool(self, call) -> str:
        if call.function.name == "search_catalog":
            arguments = _safe_json(call.function.arguments)
            if arguments is None or "query" not in arguments:
                return json.dumps({"results": [], "error": "invalid arguments"})
            return await run_search_catalog(
                self._client, self._settings, self._store, arguments
            )
        return json.dumps({"error": f"unknown tool {call.function.name}"})

    def _build_response(self, arguments: dict) -> ChatResponse:
        recommendations: list[Recommendation] = []
        seen: set[str] = set()
        for record_id in arguments.get("recommendation_ids", []):
            record = self._store.get(str(record_id))
            if record is None or record.id in seen:
                continue
            seen.add(record.id)
            recommendations.append(
                Recommendation(name=record.name, url=record.url, test_type=record.test_type)
            )
            if len(recommendations) >= self._settings.max_recommendations:
                break
        return ChatResponse(
            reply=str(arguments.get("reply", "")),
            recommendations=recommendations,
            end_of_conversation=bool(arguments.get("end_of_conversation", False)),
        )


def _assistant_dict(message) -> dict:
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in (message.tool_calls or [])
        ],
    }


def _retry_after(error: RateLimitError, default: float) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(error))
    return float(match.group(1)) + 1.0 if match else default


def _safe_json(raw: str | None) -> dict | None:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
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
    recommendations=None,
    end_of_conversation=False,
)


class Agent:
    def __init__(self, client: AsyncOpenAI, settings: Settings, store: VectorStore) -> None:
        self._client = client
        self._settings = settings
        self._store = store

    async def respond(self, messages: list[Message]) -> ChatResponse:
        for _ in range(MAX_GENERATION_RETRIES):
            response = await self._run_turn(messages)
            if response is not None:
                return response
        return _REFUSAL_FALLBACK

    async def _run_turn(self, messages: list[Message]) -> ChatResponse | None:
        system_prompt = SYSTEM_PROMPT + _turn_note(messages)
        conversation: list[dict] = [{"role": "system", "content": system_prompt}]
        conversation.extend(_history_dict(m) for m in messages if m.content.strip())

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
            reply=_strip_id_references(str(arguments.get("reply", ""))),
            recommendations=recommendations or None,
            end_of_conversation=bool(arguments.get("end_of_conversation", False)),
        )


def _strip_id_references(reply: str) -> str:
    return re.sub(r"\s*\(\s*id\s*[:#=]?\s*[\w-]+\s*\)", "", reply, flags=re.IGNORECASE)


def _history_dict(message: Message) -> dict:
    if message.role == "assistant":
        return {"role": "assistant", "content": _flatten_assistant_content(message.content)}
    return {"role": "user", "content": message.content}


def _flatten_assistant_content(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if not isinstance(parsed, dict) or "reply" not in parsed:
        return content
    text = str(parsed.get("reply", ""))
    names = [
        str(item["name"])
        for item in parsed.get("recommendations") or []
        if isinstance(item, dict) and item.get("name")
    ]
    if names:
        text += "\n\n[Shortlist recommended this turn: " + "; ".join(names) + "]"
    return text


def _turn_note(messages: list[Message]) -> str:
    user_turns = max(1, sum(1 for m in messages if m.role == "user" and m.content.strip()))
    note = f"\n\nCurrent state: this is user turn {user_turns} of at most 4."
    if user_turns >= 3:
        note += (
            " You are running low on turns — commit to a shortlist now with the constraints you"
            " have instead of asking another clarifying question."
        )
    return note


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
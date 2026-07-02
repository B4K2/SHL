import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

KNOWN_ROLES = {"user", "assistant", "system"}


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = "user"
    content: str = ""

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, value: Any) -> str:
        role = str(value).strip().lower() if value is not None else "user"
        return role if role in KNOWN_ROLES else "user"

    @field_validator("content", mode="before")
    @classmethod
    def coerce_content(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: list[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] | None = None
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"

import json
import re
from pathlib import Path

from pydantic import BaseModel

TEST_TYPE_CODES: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

TEST_TYPE_PRIORITY: list[str] = [
    "Knowledge & Skills",
    "Ability & Aptitude",
    "Simulations",
    "Biodata & Situational Judgment",
    "Competencies",
    "Personality & Behavior",
    "Development & 360",
    "Assessment Exercises",
]


class CatalogRecord(BaseModel):
    id: str
    name: str
    url: str
    test_type: str
    keys: list[str]
    description: str
    job_levels: list[str]
    languages: list[str]
    duration: str
    remote: bool
    adaptive: bool

    def embedding_text(self) -> str:
        return f"{self.name}\n\n{self.description}"

    def duration_minutes(self) -> int | None:
        match = re.search(r"(\d+)", self.duration)
        return int(match.group(1)) if match else None


def _is_job_solution(raw: dict) -> bool:
    return raw.get("name", "").strip().lower().endswith("solution")


def _resolve_test_type(keys: list[str]) -> str:
    for category in TEST_TYPE_PRIORITY:
        if category in keys:
            return TEST_TYPE_CODES[category]
    return ""


def _normalize(raw: dict) -> CatalogRecord:
    keys = raw.get("keys", []) or []
    return CatalogRecord(
        id=str(raw.get("entity_id", "")),
        name=raw.get("name", "").strip(),
        url=raw.get("link", "").strip(),
        test_type=_resolve_test_type(keys),
        keys=keys,
        description=(raw.get("description") or "").strip(),
        job_levels=raw.get("job_levels", []) or [],
        languages=raw.get("languages", []) or [],
        duration=(raw.get("duration") or "").strip(),
        remote=str(raw.get("remote", "")).strip().lower() == "yes",
        adaptive=str(raw.get("adaptive", "")).strip().lower() == "yes",
    )


def load_records(catalog_path: str | Path) -> list[CatalogRecord]:
    raw_entries = json.loads(Path(catalog_path).read_text(encoding="utf-8"), strict=False)
    return [_normalize(e) for e in raw_entries if not _is_job_solution(e)]

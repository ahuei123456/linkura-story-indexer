import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MANIFEST_SCHEMA_VERSION = "1"
SUMMARY_CACHE_SCHEMA_VERSION = "1"
RAW_EVIDENCE_SCHEMA_VERSION = "3"


class ChunkerConfig(BaseModel):
    min_chars: int
    target_chars: int
    max_chars: int


class VectorIds(BaseModel):
    raw: list[str] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)


class IngestionManifest(BaseModel):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    timestamp: str
    source_file_hashes: dict[str, str]
    parser_version: str
    chunker_version: str
    chunker_config: ChunkerConfig
    summarization_prompt_version: str
    glossary_hash: str
    chat_model: str
    generation_provider: str = "google"
    generation_model: str = ""
    embedding_model: str
    raw_evidence_schema_version: str = RAW_EVIDENCE_SCHEMA_VERSION
    summary_cache_schema_version: str = SUMMARY_CACHE_SCHEMA_VERSION
    vector_ids: VectorIds = Field(default_factory=VectorIds)


class SummaryCacheContext(BaseModel):
    source_file_hashes: dict[str, str] = Field(default_factory=dict)
    parser_version: str
    summarization_prompt_version: str
    glossary_hash: str
    chat_model: str
    generation_provider: str = "google"
    generation_model: str = ""
    embedding_model: str
    summary_cache_schema_version: str = SUMMARY_CACHE_SCHEMA_VERSION


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: list[Path]) -> dict[str, str]:
    return {str(path): hash_file(path) for path in sorted(paths, key=lambda item: str(item))}


def hash_json_file(path: Path) -> str:
    if not path.exists():
        return stable_hash(None)
    with path.open(encoding="utf-8") as file:
        return stable_hash(json.load(file))


def write_manifest(path: str | Path, manifest: IngestionManifest) -> None:
    Path(path).write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

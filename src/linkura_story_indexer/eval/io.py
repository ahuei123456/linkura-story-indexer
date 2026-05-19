import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from linkura_story_indexer.eval.models import EvalRun, GoldenSet, QueryTrace


def stable_json(data: Any) -> str:
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_golden_set(path: str | Path) -> GoldenSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"schema_version": "1", "questions": data}
    return GoldenSet.model_validate(data)


def load_eval_run(path: str | Path) -> EvalRun:
    return EvalRun.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_eval_run(run: EvalRun, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(stable_json(run), encoding="utf-8")


def write_query_trace(trace: QueryTrace, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(stable_json(trace), encoding="utf-8")

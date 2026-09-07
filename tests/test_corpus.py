from pathlib import Path

from linkura_story_indexer.indexer.corpus import prepare_corpus
from linkura_story_indexer.story_order import load_story_order


def _write(path: Path, text: str = "narrative") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_prepare_corpus_uses_explicit_root_and_manifest_order(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    first = _write(root / "103" / "episode" / "1.md", "old")
    second = _write(root / "104" / "episode" / "1.md", "new")
    config = tmp_path / "story_order.yaml"
    config.write_text(
        """
chronological_order:
  - story_type: Other
    arcs: ["104", "103"]
summary_order:
  - story_type: Other
    arcs: ["104", "103"]
""",
        encoding="utf-8",
    )
    story_order = load_story_order(config, story_root=root)

    prepared = prepare_corpus(root, story_order=story_order)

    assert prepared.files == (first, second)
    assert all(node.metadata.file_path.startswith(str(root)) for node in prepared.raw_nodes)
    assert {
        node.metadata.arc_id: node.metadata.story_order for node in prepared.raw_nodes
    } == {"104": 1, "103": 2}


def test_prepare_corpus_rejects_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    try:
        prepare_corpus(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("prepare_corpus should reject a missing story root")

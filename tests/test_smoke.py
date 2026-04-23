import importlib


def test_package_imports() -> None:
    importlib.import_module("linkura_story_indexer")

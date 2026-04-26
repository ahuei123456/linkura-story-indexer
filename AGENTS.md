# AGENTS.md

## Python Repo Policy

- Manage dependencies and tool execution with `uv`.
- Every task that modifies source files must run `uv run ruff check . --fix`, `uv run pyrefly check .`, and `uv run pytest` before the task is declared complete.
- Do not declare a source-modifying task complete unless all three commands finish with zero errors.
- Keep Ruff, Pyrefly, pytest, and prek installed through `uv` and run them with `uv run ...`.
- When running tests, it is ok to request elevated permissions as stuff will need to be done with the cache.


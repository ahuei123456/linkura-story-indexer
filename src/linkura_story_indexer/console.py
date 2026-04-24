import sys

from rich.console import Console

console = Console()


def safe_print(message: object = "") -> None:
    encoding = sys.stdout.encoding or "utf-8"
    text = str(message).encode(encoding, errors="replace").decode(encoding, errors="replace")
    console.print(text, highlight=False, markup=False)

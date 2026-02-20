from __future__ import annotations

from pathlib import Path


def write_cli_report(
    *,
    rendered: str,
    output_path: str | None,
    always_print: bool = False,
) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    if always_print or not output_path:
        print(rendered)

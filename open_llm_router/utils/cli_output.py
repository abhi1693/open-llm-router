from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable


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
        sys.stdout.write(rendered + "\n")


def render_yaml(payload: Any) -> str:
    return yaml.safe_dump(payload, sort_keys=False).rstrip()


def print_yaml(payload: Any) -> None:
    sys.stdout.write(render_yaml(payload) + "\n")


def emit_or_persist_yaml(
    *,
    payload: Any,
    dry_run: bool,
    persist: Callable[[Any], None],
) -> None:
    if dry_run:
        print_yaml(payload)
    else:
        persist(payload)

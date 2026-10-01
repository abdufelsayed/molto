"""Shared terminal presentation and machine-readable command output."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import nullcontext

from rich.console import Console
from rich.table import Table
from rich.text import Text

from molto_cli.client import CLIError


def _cell(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class Output:
    def __init__(self, args):
        self.json = getattr(args, "json", False)
        self.yes = getattr(args, "yes", False)
        self.console = Console(
            no_color=getattr(args, "no_color", False) or "NO_COLOR" in os.environ,
            highlight=False,
        )
        self.errors = Console(
            stderr=True, no_color=self.console.no_color, highlight=False
        )

    def _render(self, data, title=None):
        if isinstance(data, str):
            self.console.print(Text(data))
            return
        if isinstance(data, list):
            if not data:
                self.console.print(Text("No entries."))
                return
            if all(isinstance(row, dict) for row in data):
                columns = list(dict.fromkeys(key for row in data for key in row))
                table = Table(
                    title=title, box=None, header_style="bold cyan", padding=(0, 2)
                )
                for column in columns:
                    table.add_column(
                        column.replace("_", " ").capitalize(), overflow="fold"
                    )
                for row in data:
                    table.add_row(*(Text(_cell(row.get(column))) for column in columns))
                self.console.print(table)
                return
            for entry in data:
                self.console.print(Text(_cell(entry)))
            return
        if isinstance(data, dict):
            scalars = {
                key: value
                for key, value in data.items()
                if not isinstance(value, (dict, list))
            }
            if scalars:
                table = Table(title=title, box=None, show_header=False, padding=(0, 2))
                table.add_column(style="bold", overflow="fold")
                table.add_column(overflow="fold")
                for key, value in scalars.items():
                    table.add_row(
                        Text(key.replace("_", " ").capitalize()), Text(_cell(value))
                    )
                self.console.print(table)
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    self.console.print(
                        Text(key.replace("_", " ").capitalize(), style="bold cyan")
                    )
                    self._render(value)
            return
        self.console.print(Text(_cell(data)))

    def emit(self, data, title=None):
        if self.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            self._render(data, title)

    def error(self, error: CLIError):
        if self.json:
            print(
                json.dumps(
                    {
                        "error": {
                            "message": str(error),
                            "exit_code": error.exit_code,
                            "details": error.details,
                        }
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        else:
            self.errors.print(Text(f"Error: {error}", style="red"))

    def confirm(self, message):
        if self.yes:
            return
        if not sys.stdin.isatty() or self.json:
            raise CLIError(
                "This operation needs confirmation. Review it interactively or pass --yes.",
                2,
            )
        try:
            answer = input(f"{message} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt) as exc:
            raise CLIError("Operation cancelled.", 130) from exc
        if answer not in ("y", "yes"):
            raise CLIError("Operation cancelled.", 130)

    def watch(self, label):
        return (
            self.console.status(Text(label))
            if not self.json and self.console.is_terminal
            else nullcontext()
        )


class CLIParser(argparse.ArgumentParser):
    """Keep argparse's stable parsing and add restrained color to help."""

    def print_help(self, file=None):
        file = file or sys.stdout
        console = Console(
            file=file,
            no_color="--no-color" in sys.argv or "NO_COLOR" in os.environ,
            highlight=False,
        )
        text = Text(self.format_help())
        for line in text.plain.splitlines():
            if line.endswith(":"):
                text.highlight_regex(re.escape(line), style="bold cyan")
        console.print(text, end="", soft_wrap=True)

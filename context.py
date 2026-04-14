"""Manages local file/directory context that gets injected into Bedrock requests."""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path

# Files/dirs to skip when loading a directory
IGNORE_PATTERNS = [
    ".git", ".svn", "__pycache__", "*.pyc", "*.pyo",
    "node_modules", ".venv", "venv", "env",
    "*.egg-info", "dist", "build", ".mypy_cache", ".pytest_cache",
    ".DS_Store", "*.log", "*.lock", "package-lock.json",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg",
    "*.pdf", "*.zip", "*.tar", "*.gz", "*.bin", "*.exe",
]

MAX_FILE_BYTES = 100_000   # 100 KB per file
MAX_TOTAL_BYTES = 500_000  # 500 KB total context


def _should_ignore(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in IGNORE_PATTERNS)


def _read_file(path: Path) -> str | None:
    """Read a file as text, returning None if it's binary or too large."""
    if path.stat().st_size > MAX_FILE_BYTES:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, PermissionError):
        return None


class FileContext:
    def __init__(self) -> None:
        # label -> content  (ordered)
        self._files: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path_str: str, base_dir: str | None = None) -> list[str]:
        """
        Load a file or directory. Returns a list of messages describing
        what was loaded (or skipped).
        """
        path = Path(path_str).expanduser()
        if not path.is_absolute() and base_dir:
            path = Path(base_dir) / path
        path = path.resolve()

        if not path.exists():
            return [f"Path not found: {path}"]

        if path.is_file():
            return self._load_file(path)

        if path.is_dir():
            return self._load_dir(path)

        return [f"Not a file or directory: {path}"]

    def unload(self) -> int:
        n = len(self._files)
        self._files.clear()
        return n

    def unload_path(self, label: str) -> bool:
        if label in self._files:
            del self._files[label]
            return True
        return False

    def list_loaded(self) -> list[tuple[str, int]]:
        """Return [(label, char_count), ...]"""
        return [(k, len(v)) for k, v in self._files.items()]

    def get_context_block(self) -> str:
        """Return a formatted string to inject into the system prompt."""
        if not self._files:
            return ""
        parts = ["The following local files have been provided for context:\n"]
        for label, content in self._files.items():
            parts.append(f"<file path=\"{label}\">\n{content}\n</file>")
        return "\n".join(parts)

    def total_chars(self) -> int:
        return sum(len(v) for v in self._files.values())

    def resolve_at_references(self, text: str, base_dir: str | None = None) -> tuple[str, list[str]]:
        """
        Replace @filename tokens in text with inline file content.
        Returns (expanded_text, [messages]).
        """
        import re
        messages: list[str] = []
        pattern = re.compile(r"@([\w./_\-]+)")

        def replace(m: re.Match) -> str:
            ref = m.group(1)
            msgs = self.load(ref, base_dir)
            messages.extend(msgs)
            if ref in self._files or any(ref in k for k in self._files):
                return ""   # file loaded, will appear in context block
            return m.group(0)

        expanded = pattern.sub(replace, text)
        return expanded.strip(), messages

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> list[str]:
        label = str(path)
        content = _read_file(path)
        if content is None:
            return [f"Skipped (binary or too large): {path.name}"]

        if self.total_chars() + len(content) > MAX_TOTAL_BYTES:
            return [f"Skipped (context budget exceeded): {path.name}"]

        self._files[label] = content
        return [f"Loaded: {path.name}  ({len(content):,} chars)"]

    def _load_dir(self, root: Path) -> list[str]:
        messages: list[str] = []
        loaded = 0
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune ignored directories in-place so os.walk skips them
            dirnames[:] = [d for d in sorted(dirnames) if not _should_ignore(d)]
            for fname in sorted(filenames):
                if _should_ignore(fname):
                    continue
                fpath = Path(dirpath) / fname
                msgs = self._load_file(fpath)
                messages.extend(msgs)
                if msgs and msgs[0].startswith("Loaded"):
                    loaded += 1
                if self.total_chars() >= MAX_TOTAL_BYTES:
                    messages.append("[dim]Context budget reached — remaining files skipped.[/dim]")
                    return messages
        messages.append(f"Directory loaded: {loaded} file(s) from {root}")
        return messages

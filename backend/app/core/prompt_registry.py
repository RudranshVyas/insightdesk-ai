"""Versioned prompt registry.

A prompt is code. Changing the wording changes the output, which invalidates
every measured number produced under the old wording. So prompts live in
versioned directories on disk, each with metadata, and every evaluation record
and Support Brief stamps the version it used.

The rule: **never edit a shipped version in place.** Add `v2/` and switch the
default. `content_sha256` exists so an accidental in-place edit is detectable
rather than silent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class PromptNotFound(Exception):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    user_template: str | None
    metadata: dict[str, Any]
    content_sha256: str

    @property
    def stamp(self) -> str:
        """What goes into an evaluation record: name, version, content hash."""
        return f"{self.name}/{self.version}@{self.content_sha256[:12]}"

    def render_user(self, **kwargs: Any) -> str:
        if self.user_template is None:
            raise PromptNotFound(f"{self.name}/{self.version} has no user template")
        required = set(self.metadata.get("user_template_variables") or [])
        missing = required - set(kwargs)
        if missing:
            raise ValueError(
                f"{self.stamp}: missing template variables {sorted(missing)}. "
                f"A silently unfilled placeholder would ship '{{{sorted(missing)[0]}}}' "
                f"to the provider as literal text."
            )
        return self.user_template.format(**kwargs)

    def generation_settings(self) -> dict[str, Any]:
        return dict(self.metadata.get("generation_settings") or {})


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str = "v1", root: str | None = None) -> Prompt:
    base = Path(root) if root else PROMPTS_DIR
    directory = base / name
    if not directory.is_dir():
        raise PromptNotFound(f"no prompt directory at {directory}")

    system = _read(directory / f"{version}_system.txt")
    if system is None:
        raise PromptNotFound(f"{name}/{version}: missing {version}_system.txt")
    user = _read(directory / f"{version}_user.txt")

    meta_raw = _read(directory / "metadata.json")
    metadata = json.loads(meta_raw) if meta_raw else {}
    if metadata.get("version") and metadata["version"] != version:
        raise ValueError(
            f"{name}/{version}: metadata.json declares version "
            f"{metadata['version']!r}. Each version needs its own metadata."
        )

    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    if user:
        h.update(user.encode("utf-8"))

    return Prompt(name, version, system, user, metadata, h.hexdigest())


def list_prompts(root: str | None = None) -> list[dict[str, Any]]:
    base = Path(root) if root else PROMPTS_DIR
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        versions = sorted(
            f.name.split("_", 1)[0] for f in directory.glob("*_system.txt")
        )
        for version in versions:
            try:
                p = load_prompt(directory.name, version, root)
            except (PromptNotFound, ValueError):
                continue
            out.append(
                {
                    "name": p.name,
                    "version": p.version,
                    "stamp": p.stamp,
                    "purpose": p.metadata.get("purpose"),
                }
            )
    return out

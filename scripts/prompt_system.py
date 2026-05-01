from __future__ import annotations

"""Shared prompt engineering utilities for SAM3 scripts.

This module centralizes prompt normalization, alias resolution, attribute-based
prompt expansion, and candidate generation so that training and evaluation
scripts can use the same logic instead of hand-tuning prompts case by case.
"""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ATTRIBUTES_JSON = SCRIPT_DIR / "prompt_templates" / "attributes_template.json"
DEFAULT_ALIASES_JSON = SCRIPT_DIR / "prompt_templates" / "aliases_template.json"


def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s_\-]+", " ", s)
    return s


def load_json(path: Path | None) -> dict | list:
    if path is None:
        return {}
    return json.loads(path.read_text())


@dataclass(frozen=True)
class PromptExpansion:
    raw_prompt: str
    canonical_prompt: str | None
    expanded_prompt: str
    attrs_used: list[str]
    matched: bool
    source: str


class PromptSystem:
    """Central prompt registry for aliasing, canonicalization, and expansion."""

    def __init__(
        self,
        attributes: dict[str, list[str]] | None = None,
        aliases: dict[str, str] | None = None,
        attributes_json_path: Path | None = None,
        aliases_json_path: Path | None = None,
    ):
        self.attributes = attributes or {}
        self.aliases = aliases or {}
        self.attributes_json_path = attributes_json_path
        self.aliases_json_path = aliases_json_path
        self._canonical_norm_to_canonical = {
            norm_text(name): name for name in self.attributes.keys()
        }
        self._alias_norm_to_canonical = {
            norm_text(str(alias)): str(canonical)
            for alias, canonical in self.aliases.items()
        }

    @classmethod
    def from_paths(
        cls,
        attributes_json: Path | None = None,
        aliases_json: Path | None = None,
    ) -> "PromptSystem":
        attributes_path = attributes_json or DEFAULT_ATTRIBUTES_JSON
        aliases_path = aliases_json or DEFAULT_ALIASES_JSON

        attributes: dict[str, list[str]] = {}
        aliases: dict[str, str] = {}

        if attributes_path is not None and attributes_path.exists():
            payload = load_json(attributes_path)
            if not isinstance(payload, dict):
                raise ValueError("attributes-json must be a JSON object")
            attributes = {
                str(key): [str(item) for item in value]
                for key, value in payload.items()
            }

        if aliases_path is not None and aliases_path.exists():
            payload = load_json(aliases_path)
            if not isinstance(payload, dict):
                raise ValueError("aliases-json must be a JSON object")
            aliases = {str(key): str(value) for key, value in payload.items()}

        return cls(
            attributes=attributes,
            aliases=aliases,
            attributes_json_path=attributes_path,
            aliases_json_path=aliases_path,
        )

    def is_enabled(self) -> bool:
        return len(self.attributes) > 0 or len(self.aliases) > 0

    def resolve_canonical(self, raw_prompt: str) -> str | None:
        key = norm_text(raw_prompt)
        if key in self._alias_norm_to_canonical:
            return self._alias_norm_to_canonical[key]
        if key in self._canonical_norm_to_canonical:
            return self._canonical_norm_to_canonical[key]
        return None

    def compose_prompt(
        self,
        canonical: str,
        attrs: list[str],
        fmt: str = "attrs_then_class",
        separator: str = ", ",
    ) -> str:
        attrs = [attr.strip() for attr in attrs if str(attr).strip()]
        if not attrs:
            return canonical
        if fmt == "class_then_attrs":
            return separator.join([canonical] + attrs)
        return separator.join(attrs + [canonical])

    def expand_prompt(
        self,
        raw_prompt: str,
        topk_attrs: int = 3,
        fmt: str = "attrs_then_class",
        separator: str = ", ",
        unknown_policy: str = "keep_raw",
    ) -> PromptExpansion:
        canonical = self.resolve_canonical(raw_prompt)
        if canonical is None:
            if unknown_policy == "error":
                raise KeyError(f"Unknown prompt: {raw_prompt!r}")
            return PromptExpansion(
                raw_prompt=raw_prompt,
                canonical_prompt=None,
                expanded_prompt=raw_prompt,
                attrs_used=[],
                matched=False,
                source="raw",
            )

        attrs = self.attributes.get(canonical, [])
        if topk_attrs > 0:
            attrs = attrs[:topk_attrs]
        expanded = self.compose_prompt(canonical, attrs, fmt=fmt, separator=separator)
        return PromptExpansion(
            raw_prompt=raw_prompt,
            canonical_prompt=canonical,
            expanded_prompt=expanded,
            attrs_used=attrs,
            matched=True,
            source="expanded" if expanded != canonical else "canonical",
        )

    def transform_prompt(
        self,
        raw_prompt: str,
        mode: str = "raw",
        topk_attrs: int = 3,
        fmt: str = "attrs_then_class",
        separator: str = ", ",
    ) -> str:
        if mode == "raw":
            return raw_prompt
        expansion = self.expand_prompt(
            raw_prompt=raw_prompt,
            topk_attrs=topk_attrs,
            fmt=fmt,
            separator=separator,
        )
        if mode == "canonical":
            return expansion.canonical_prompt or raw_prompt
        if mode == "expanded":
            return expansion.expanded_prompt
        raise ValueError(f"Unsupported prompt mode: {mode}")

    def candidate_prompts(
        self,
        raw_prompt: str,
        topk_attrs: int = 3,
        fmt: str = "attrs_then_class",
        separator: str = ", ",
        include_raw: bool = True,
        include_canonical: bool = True,
        include_expanded: bool = True,
    ) -> list[PromptExpansion]:
        expansions: list[PromptExpansion] = []
        seen: set[str] = set()

        def add(candidate: PromptExpansion):
            key = norm_text(candidate.expanded_prompt)
            if key in seen:
                return
            seen.add(key)
            expansions.append(candidate)

        if include_raw:
            add(
                PromptExpansion(
                    raw_prompt=raw_prompt,
                    canonical_prompt=self.resolve_canonical(raw_prompt),
                    expanded_prompt=raw_prompt,
                    attrs_used=[],
                    matched=self.resolve_canonical(raw_prompt) is not None,
                    source="raw",
                )
            )

        canonical = self.resolve_canonical(raw_prompt)
        if canonical is not None and include_canonical:
            add(
                PromptExpansion(
                    raw_prompt=raw_prompt,
                    canonical_prompt=canonical,
                    expanded_prompt=canonical,
                    attrs_used=[],
                    matched=True,
                    source="canonical",
                )
            )

        if include_expanded:
            add(self.expand_prompt(raw_prompt, topk_attrs, fmt, separator))

        return expansions

    def manifest(self) -> dict:
        return {
            "enabled": self.is_enabled(),
            "num_attributes": len(self.attributes),
            "num_aliases": len(self.aliases),
            "attributes_json": None if self.attributes_json_path is None else str(self.attributes_json_path),
            "aliases_json": None if self.aliases_json_path is None else str(self.aliases_json_path),
        }

    def expand_json_payload(
        self,
        payload: dict | list,
        topk_attrs: int = 3,
        unknown_policy: str = "keep_raw",
        fmt: str = "attrs_then_class",
        separator: str = ", ",
    ):
        if isinstance(payload, dict):
            output = {}
            details = {}
            known = 0
            unknown = 0
            for key, raw_prompt in payload.items():
                if raw_prompt is None:
                    output[key] = None
                    details[key] = {
                        "raw_prompt": None,
                        "expanded_prompt": None,
                        "matched": False,
                    }
                    unknown += 1
                    continue
                expansion = self.expand_prompt(
                    str(raw_prompt),
                    topk_attrs=topk_attrs,
                    fmt=fmt,
                    separator=separator,
                    unknown_policy=unknown_policy,
                )
                output[key] = expansion.expanded_prompt
                details[key] = {
                    "raw_prompt": expansion.raw_prompt,
                    "expanded_prompt": expansion.expanded_prompt,
                    "attrs_used": expansion.attrs_used,
                    "matched": expansion.matched,
                }
                if expansion.matched:
                    known += 1
                else:
                    unknown += 1
            return output, details, known, unknown

        if isinstance(payload, list):
            output = []
            details = []
            known = 0
            unknown = 0
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    raise ValueError(f"datasets-json[{idx}] must be an object")
                if "prompt" not in item:
                    raise ValueError(f"datasets-json[{idx}] missing prompt field")
                expansion = self.expand_prompt(
                    str(item["prompt"]),
                    topk_attrs=topk_attrs,
                    fmt=fmt,
                    separator=separator,
                    unknown_policy=unknown_policy,
                )
                new_item = dict(item)
                new_item["prompt_raw"] = expansion.raw_prompt
                new_item["prompt"] = expansion.expanded_prompt
                output.append(new_item)
                details.append(
                    {
                        "index": idx,
                        "raw_prompt": expansion.raw_prompt,
                        "expanded_prompt": expansion.expanded_prompt,
                        "attrs_used": expansion.attrs_used,
                        "matched": expansion.matched,
                    }
                )
                if expansion.matched:
                    known += 1
                else:
                    unknown += 1
            return output, details, known, unknown

        raise ValueError("input-json must be either a JSON object or a JSON list")

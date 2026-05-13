#!/usr/bin/env python3
"""
Adaptive PLATO Formatter — adjusts PLATO room structure based on target model capability.

Profiles:
  - tiny  (≤1B params):  Strip all tags, return plain text
  - mid   (1-30B params): Full PLATO structure with domain tags, keys, cross-refs, warnings
  - large (30B+ params): Keys only, minimal structure, no domain tags or cross-refs

Usage:
    python3 -m adaptive_plato --model seed-2.0-mini --query "Eisenstein snap" rooms.json
    python3 -m adaptive_plato --detect "qwen3:0.6b"
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class ModelProfile:
    """Well-known model profile constants."""
    TINY = "tiny"     # ≤1B params: strip all structure
    MID = "mid"       # 1-30B params: full structure
    LARGE = "large"   # 30B+ params: minimal structure (keys only)

    ALL = (TINY, MID, LARGE)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoredRoom:
    """A PLATO room with a relevance score."""
    room: dict
    score: float = 0.0

    def __lt__(self, other: "ScoredRoom") -> bool:
        return self.score < other.score


# ---------------------------------------------------------------------------
# Auto-detection patterns
# ---------------------------------------------------------------------------

# Pattern: (regex, profile)
# Ordered most-specific to most-general.
_DETECT_PATTERNS: list[tuple[re.Pattern, str]] = []


def _compile_patterns():
    """Build detection patterns — called once at module load."""
    if _DETECT_PATTERNS:
        return

    # tiny (≤1B) — specific small variants
    tiny_patterns = [
        re.compile(r"qwen3?[:\-.]?0\.?\s*6b?", re.I),
        re.compile(r"llama3?\.[12][:\-.]?1b?", re.I),
        re.compile(r"gemma[:\-.]?0\.[25]b?", re.I),
        re.compile(r"tinyllama", re.I),
        re.compile(r"smollm", re.I),
        re.compile(r"phi[:\-.]?1\b", re.I),
        re.compile(r"stablelm[:\-.]?1b", re.I),
    ]
    # mid (1-30B)
    mid_patterns = [
        re.compile(r"glm[:\-.]?5[:\-.]?turbo", re.I),
        re.compile(r"glm[:\-.]?4[:\-.]?7", re.I),  # 8B
        re.compile(r"glm[:\-.]?4[:\-.]?5", re.I),
        re.compile(r"mistral[:\-.]?7b", re.I),
        re.compile(r"llama3?\.[123][:\-.]?[38]b?", re.I),
        re.compile(r"llama[:\-.]?3[:\-.]?1[:\-.]?8b", re.I),
        re.compile(r"mixtral[:\-.]?8x7b", re.I),
        re.compile(r"qwen[:\-.]?2[:\-.]?5[:\-.]?7b", re.I),
        re.compile(r"deepseek[:\-.]?v2", re.I),
        re.compile(r"deepseek[:\-.]?coder[:\-.]?[67]b", re.I),
        re.compile(r"gemma[:\-.]?2[:\-.]?[279]b?", re.I),
        re.compile(r"gemma[:\-.]?3[:\-.]?[148]b?", re.I),
        re.compile(r"phi[:\-.]?3[:\-]?(mini|small|medium)", re.I),
        re.compile(r"codestral", re.I),
        re.compile(r"codegemma", re.I),
        re.compile(r"starcoder[:\-.]?[137]b", re.I),
        re.compile(r"starcoder2[:\-.]?[137]b", re.I),
        re.compile(r"codeqwen[:\-.]?7b", re.I),
    ]
    # large (30B+) — check before mid because some large contain "llama" too
    large_patterns = [
        re.compile(r"hermes.*?(?:405|70)\s*B?", re.I),
        re.compile(r"llama.*?[0-9]{2,3}\s*B?", re.I),
        re.compile(r"llama[:\-._]?4[:\-._]?1[:\-._]?405b", re.I),
        re.compile(r"seed[:\-.]?2[:\-.]?0[:\-.]?(mini|pro|code)", re.I),
        re.compile(r"deepseek[:\-.]?v3", re.I),
        re.compile(r"deepseek[:\-.]?r1", re.I),
        re.compile(r"qwen[:\-.]?2[:\-.]?[57]2b", re.I),
        re.compile(r"qwen3.*?[0-9]{2,3}\s*B?", re.I),  # catch all Qwen3 variants e.g. Qwen3-235B
        re.compile(r"qwen[:\-.]?2[:\-.]?[57]2b", re.I),    # 32B+, Qwen2 72B
        re.compile(r"command[:\-]?r[:\-]?(plus)?", re.I),
        re.compile(r"deepseek[:\-.]?chat", re.I),
        re.compile(r"claude[:\-.]?[34]\.", re.I),
        re.compile(r"gpt[:\-]?4", re.I),
        re.compile(r"grok", re.I),
        re.compile(r"gemini[:\-.]?1\.5", re.I),
        re.compile(r"gemini[:\-.]?2\.", re.I),
        re.compile(r"yi[:\-.]?34b", re.I),
        re.compile(r"dbrx", re.I),
        re.compile(r"mixtral[:\-.]?8x22b", re.I),
        re.compile(r"nemotron", re.I),
    ]

    for p in tiny_patterns:
        _DETECT_PATTERNS.append((p, ModelProfile.TINY))
    for p in large_patterns:
        _DETECT_PATTERNS.append((p, ModelProfile.LARGE))
    for p in mid_patterns:
        _DETECT_PATTERNS.append((p, ModelProfile.MID))


_compile_patterns()


def _detect_profile_from_name(model_name: str) -> str | None:
    """Return profile string if model_name matches a known pattern, else None."""
    for pattern, profile in _DETECT_PATTERNS:
        if pattern.search(model_name):
            return profile
    return None


def _fallback_detect(model_name: str) -> str | None:
    """Guess from size indicators embedded in the name."""
    # Look for 'B' or 'b' suffix indicating param count
    m = re.search(r'[:\-_.](\d+)\s*B', model_name, re.I)
    if m:
        params = int(m.group(1))
        if params <= 1:
            return ModelProfile.TINY
        elif params <= 30:
            return ModelProfile.MID
        else:
            return ModelProfile.LARGE

    # Look for 'b' (lowercase) — likely means billions in model naming
    m = re.search(r'[:\-_.](\d+)\s*b', model_name, re.I)
    if m:
        params = int(m.group(1))
        if params <= 1:
            return ModelProfile.TINY
        elif params <= 30:
            return ModelProfile.MID
        else:
            return ModelProfile.LARGE

    return None


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def _extract_plain_text(room_data: dict) -> str:
    """Extract content as plain text, stripping ALL structural metadata."""
    lines = []

    # Title is always useful
    title = _safe_str(room_data.get("title", room_data.get("name", "")))
    if title:
        lines.append(title)

    # Content — strip tags from each line
    content = room_data.get("content", room_data.get("body", ""))
    if isinstance(content, str):
        for line in content.split("\n"):
            cleaned = _strip_tags(line)
            if cleaned.strip():
                lines.append(cleaned)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                cleaned = _strip_tags(item)
                if cleaned.strip():
                    lines.append(cleaned)
            elif isinstance(item, dict):
                cleaned = _strip_tags(_safe_str(item.get("text", "")))
                if cleaned.strip():
                    lines.append(cleaned)

    return "\n".join(lines)


_TAG_RE = re.compile(
    r"\[(KEY|CROSS-REF|WARNING|DOMAIN|TOPIC|STATUS|PRIORITY|BLOCKER|DECISION|NOTE|META|SOURCE)"
    r":.*?\]",
    re.I,
)


def _strip_tags(text: str) -> str:
    """Remove PLATO structural tags like [KEY:xxx], [WARNING:xxx]."""
    return _TAG_RE.sub("", text).strip()


def _format_mid(room_data: dict) -> str:
    """Full PLATO structure with all tags preserved."""
    lines = []
    title = _safe_str(room_data.get("title", room_data.get("name", "")))
    domain = _safe_str(room_data.get("domain", ""))
    tags = room_data.get("tags", [])

    if title:
        lines.append(f"# {title}")

    if domain:
        lines.append(f"[DOMAIN: {domain}]")

    if tags:
        lines.append(f"[TOPIC: {', '.join(tags[:5])}]")

    content = room_data.get("content", room_data.get("body", ""))
    if isinstance(content, str):
        lines.append("")
        lines.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, dict):
                text = _safe_str(item.get("text", ""))
                if text:
                    lines.append(text)

    return "\n".join(lines)


def _format_large(room_data: dict) -> str:
    """Keys-only, minimal structure, no domain tags or cross-refs."""
    lines = []
    title = _safe_str(room_data.get("title", room_data.get("name", "")))
    if title:
        lines.append(title)
        lines.append("=" * len(title))

    content = room_data.get("content", room_data.get("body", ""))
    if isinstance(content, str):
        # Preserve only [KEY:] tags, strip the rest
        for line in content.split("\n"):
            if "[KEY:" in line:
                lines.append(line)
            else:
                stripped = _TAG_RE.sub("", line)
                if stripped.strip():
                    lines.append(stripped)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                if "[KEY:" in item:
                    lines.append(item)
                else:
                    stripped = _TAG_RE.sub("", item)
                    if stripped.strip():
                        lines.append(stripped)
            elif isinstance(item, dict):
                text = _safe_str(item.get("text", ""))
                if "[KEY:" in text:
                    lines.append(text)
                else:
                    stripped = _TAG_RE.sub("", text)
                    if stripped.strip():
                        lines.append(stripped)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def compute_relevance(query: str, room: dict) -> float:
    """Compute relevance of a room to a query.
    
    Factors:
    - Keyword match ratio in title/content
    - Domain match bonus
    - Tag match bonus
    """
    if not query or not query.strip():
        return 0.0

    query_lower = query.lower()
    query_terms = set(query_lower.split())

    score = 0.0
    text_pool = ""

    title = _safe_str(room.get("title", room.get("name", "")))
    text_pool += title.lower() + " "

    content = room.get("content", room.get("body", ""))
    if isinstance(content, str):
        text_pool += content.lower()
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                text_pool += item.lower() + " "
            elif isinstance(item, dict):
                text_pool += _safe_str(item.get("text", "")).lower() + " "

    # Exact phrase match bonus
    if query_lower in text_pool:
        score += 10.0

    # Per-term matches
    for term in query_terms:
        score += text_pool.count(term) * 2.0

    # Domain match
    domain = _safe_str(room.get("domain", "")).lower()
    if domain and any(term in domain for term in query_terms):
        score += 5.0

    # Tag matches
    tags = room.get("tags", [])
    for tag in tags:
        tag_lower = _safe_str(tag).lower()
        if any(term in tag_lower for term in query_terms):
            score += 3.0

    return score


def compute_recency(room: dict, now: datetime | None = None) -> float:
    """Recency score — newer rooms score higher (max 5.0)."""
    if now is None:
        now = datetime.now(timezone.utc)

    updated_raw = room.get("updated", room.get("created", room.get("date", "")))
    if not updated_raw:
        return 0.0

    # Try ISO format
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            updated = datetime.strptime(
                updated_raw.replace("Z", "+0000"),
                fmt.replace("%z", "+0000") if "%z" in fmt else fmt,
            ).replace(tzinfo=timezone.utc)
            delta = now - updated
            days = delta.total_seconds() / 86400.0
            if days < 0:
                return 5.0
            return max(0.0, 5.0 * (1.0 - days / 365.0))
        except (ValueError, AttributeError):
            continue

    return 0.0


def compute_domain_density(room: dict) -> float:
    """Domain density score — more structured rooms get bonus for 'mid' profile.
    
    Returns 0-3 based on how many structural fields are present.
    """
    score = 0.0
    if room.get("domain"):
        score += 1.0
    if room.get("tags"):
        score += 1.0
    content = room.get("content", room.get("body", ""))
    if isinstance(content, str) and "[KEY:" in content:
        score += 1.0
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str) and "[KEY:" in item:
                score += 1.0
                break
    return score


def score_room(query: str, room: dict, now: datetime | None = None) -> float:
    """Combined relevance + recency + domain density score."""
    return (
        compute_relevance(query, room)
        + compute_recency(room, now)
        + compute_domain_density(room)
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AdaptiveFormatter:
    """Adjusts PLATO room structure for target model capability.

    Profiles:
      - tiny  (≤1B): Strip all tags, return plain text
      - mid   (1-30B): Full PLATO structure with domain tags, keys, cross-refs
      - large (30B+): Keys only, minimal structure, no domain tags or cross-refs
    """

    _PROFILE_TO_FN = {
        ModelProfile.TINY: _extract_plain_text,
        ModelProfile.MID: _format_mid,
        ModelProfile.LARGE: _format_large,
    }

    def __init__(self, model_profile: str = "auto"):
        if model_profile not in (*ModelProfile.ALL, "auto"):
            raise ValueError(
                f"Unknown model_profile {model_profile!r}. "
                f"Choose from {ModelProfile.ALL}, 'auto'."
            )
        self.profile = model_profile

    def detect_profile(self, model_name: str) -> str:
        """Auto-detect model profile from model name."""
        if not model_name:
            return ModelProfile.MID  # safe default

        result = _detect_profile_from_name(model_name)
        if result is not None:
            return result

        result = _fallback_detect(model_name)
        if result is not None:
            return result

        # Generic heuristic based on naming conventions
        name_lower = model_name.lower()

        # Known tiny keywords
        if any(kw in name_lower for kw in ("tiny", "mini", "small", "nano", "pico", "micro", "1b", "0.")):
            return ModelProfile.TINY

        # Known large keywords
        if any(kw in name_lower for kw in ("large", "huge", "big", "giant", "xxl", "405b", "120b", "xl")):
            return ModelProfile.LARGE

        # Param-count patterns that strongly indicate large models
        if re.search(r'(?:70|90|120|235)\s*b', name_lower, re.I):
            return ModelProfile.LARGE

        # Conservative default for unknowns: mid (structure helps most models)
        return ModelProfile.MID

    def resolve_profile(self, model_name: str | None = None) -> str:
        """Return the effective profile, auto-detected if needed."""
        if self.profile != "auto":
            return self.profile
        if model_name:
            return self.detect_profile(model_name)
        return ModelProfile.MID

    def format_room(
        self, room_data: dict, query: str = "", model_name: str | None = None
    ) -> str:
        """Format a single PLATO room for the target model.

        Args:
            room_data: Room dict with keys like 'title', 'content', 'domain', 'tags'
            query: Optional query string for context
            model_name: Optional model name for auto-detection

        Returns:
            Formatted room string
        """
        profile = self.resolve_profile(model_name)
        formatter = self._PROFILE_TO_FN.get(profile)
        if formatter is None:
            raise ValueError(f"No formatter for profile {profile!r}")

        formatted = formatter(room_data)

        # If we have a query, prepend it as context for small models that need it
        if profile == ModelProfile.TINY and query.strip():
            formatted = f"Context: {query}\n\n{formatted}"

        return formatted

    def format_rooms(
        self,
        rooms: list[dict],
        query: str = "",
        model_name: str | None = None,
    ) -> str:
        """Format multiple rooms, ordered by relevance to query.

        Args:
            rooms: List of room dicts
            query: Query string for relevance scoring and ordering
            model_name: Optional model name for auto-detection

        Returns:
            Concatenated formatted rooms, ordered by descending relevance
        """
        profile = self.resolve_profile(model_name)
        now = datetime.now(timezone.utc)

        scored: list[ScoredRoom] = []
        for room in rooms:
            if not isinstance(room, dict):
                continue
            s = score_room(query, room, now)
            scored.append(ScoredRoom(room=room, score=s))

        scored.sort(reverse=True)

        # If query is empty, return all; otherwise respect relevance
        if query.strip():
            scored = [sr for sr in scored if sr.score > 0]

        parts = []
        for i, sr in enumerate(scored, 1):
            formatted = self.format_room(sr.room, query=query, model_name=model_name)
            if profile == ModelProfile.TINY:
                parts.append(formatted)
            else:
                heading = f"--- Room {i} | score={sr.score:.1f} ---"
                width = max(len(heading), 60)
                parts.append(f"{'─' * width}")
                parts.append(f"  Room {i}  (score={sr.score:.1f})")
                parts.append(f"{'─' * width}")
                parts.append(formatted)

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive PLATO Formatter — format rooms for target model capability"
    )
    parser.add_argument(
        "rooms_file",
        nargs="?",
        default=None,
        help="Path to JSON file containing PLATO rooms (list of dicts or dict key)",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model name or profile (tiny|mid|large|auto). Default: auto",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Force profile (tiny|mid|large). Overrides auto-detection.",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Query string for relevance scoring and ordering",
    )
    parser.add_argument(
        "--detect",
        default=None,
        help="Detect profile for a model name and exit",
    )
    parser.add_argument(
        "--rooms-key",
        default=None,
        help="If rooms_file contains a dict, use this key for the room list",
    )

    args = parser.parse_args()

    formatter = AdaptiveFormatter(
        model_profile=args.profile if args.profile else "auto"
    )

    # Just detect mode
    if args.detect:
        profile = formatter.detect_profile(args.detect)
        print(f"Model: {args.detect}")
        print(f"Profile: {profile}")
        return

    # Read rooms
    if args.rooms_file is None:
        parser.print_help()
        sys.exit(1)

    path = Path(args.rooms_file)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(path.read_text())

    rooms: list[dict] = []
    if isinstance(raw, list):
        rooms = raw
    elif isinstance(raw, dict):
        if args.rooms_key:
            rooms = raw[args.rooms_key]
        else:
            rooms = raw.get("rooms", raw.get("room", raw.get("data", [])))
            if isinstance(rooms, dict):
                rooms = [rooms]
    else:
        print("Error: rooms data must be a list of dicts", file=sys.stderr)
        sys.exit(1)

    if not rooms:
        print("No rooms found in input.")
        return

    output = formatter.format_rooms(
        rooms=rooms,
        query=args.query,
        model_name=args.model if args.model != "auto" else None,
    )

    print(output)


if __name__ == "__main__":
    main()

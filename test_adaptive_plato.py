#!/usr/bin/env python3
"""Tests for adaptive_plato module — 30+ tests covering all profiles, auto-detect, scoring, edge cases."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from adaptive_plato import (
    AdaptiveFormatter,
    ModelProfile,
    _detect_profile_from_name,
    _fallback_detect,
    _extract_plain_text,
    _format_mid,
    _format_large,
    _strip_tags,
    compute_relevance,
    compute_recency,
    compute_domain_density,
    score_room,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SAMPLE_ROOM: dict[str, Any] = {
    "title": "Constraint Theory — Eisenstein Snap",
    "domain": "plato/forge/constraints",
    "tags": ["constraint-theory", "eisenstein", "verification"],
    "updated": "2026-05-12T14:30:00Z",
    "content": (
        "[KEY: Eisenstein snap] When two symmetric constraint regions collide.\n"
        "[KEY: Detection method] Monitor second derivative of constraint gradient.\n"
        "[CROSS-REF: verification/phase-transitions.md]\n"
        "[WARNING: False positives near boundary conditions.]\n"
        "[DOMAIN: constraint-theory] Core algorithmic insight.\n"
        "Additional explanatory text for context."
    ),
}

QUERY = "Eisenstein snap detection"


# ---------------------------------------------------------------------------
# _strip_tags
# ---------------------------------------------------------------------------


class TestStripTags:
    def test_strip_key_tag(self):
        assert _strip_tags("[KEY: foo] bar") == "bar"

    def test_strip_cross_ref(self):
        assert _strip_tags("[CROSS-REF: x.md] text") == "text"

    def test_strip_warning(self):
        assert _strip_tags("[WARNING: danger] keep") == "keep"

    def test_strip_domain_tag(self):
        assert _strip_tags("[DOMAIN: foo] bar baz") == "bar baz"

    def test_strip_multiple_tags(self):
        result = _strip_tags("[KEY: a] first [WARNING: b] second")
        assert "first  second" == result

    def test_no_tags_unchanged(self):
        assert _strip_tags("plain text") == "plain text"

    def test_empty_string(self):
        assert _strip_tags("") == ""

    def test_all_tags(self):
        result = _strip_tags("[KEY: x] [CROSS-REF: y] [WARNING: z] [DOMAIN: a]")
        assert result == ""


# ---------------------------------------------------------------------------
# _extract_plain_text (tiny profile)
# ---------------------------------------------------------------------------


class TestExtractPlainText:
    def test_basic_room(self):
        result = _extract_plain_text(SAMPLE_ROOM)
        # Should have title, stripped lines, no tags
        assert "Constraint Theory — Eisenstein Snap" in result
        assert "[KEY:" not in result
        assert "[CROSS-REF:" not in result
        assert "[WARNING:" not in result
        assert "[DOMAIN:" not in result
        assert "When two symmetric constraint regions collide." in result

    def test_no_content(self):
        result = _extract_plain_text({"title": "Empty"})
        assert result == "Empty"

    def test_content_as_list(self):
        room = {"title": "List Room", "content": ["[KEY: foo] line one", "[WARNING: bar] line two"]}
        result = _extract_plain_text(room)
        assert "List Room" in result
        assert "[KEY:" not in result
        assert "line one" in result
        assert "line two" in result

    def test_content_as_list_of_dicts(self):
        room = {"title": "Dicts", "content": [{"text": "[KEY: a] first"}, {"text": "[CROSS-REF: b] second"}]}
        result = _extract_plain_text(room)
        assert "first" in result
        assert "second" in result
        assert "[KEY:" not in result

    def test_missing_title_uses_name(self):
        result = _extract_plain_text({"name": "Fallback Name", "content": "body"})
        assert "Fallback Name" in result

    def test_empty_room(self):
        result = _extract_plain_text({})
        assert result == ""


# ---------------------------------------------------------------------------
# _format_mid (full structure)
# ---------------------------------------------------------------------------


class TestFormatMid:
    def test_basic_room(self):
        result = _format_mid(SAMPLE_ROOM)
        assert "# Constraint Theory — Eisenstein Snap" in result
        assert "[DOMAIN: plato/forge/constraints]" in result
        assert "[TOPIC:" in result
        assert "[KEY: Eisenstein snap]" in result
        assert "[CROSS-REF:" in result
        assert "[WARNING:" in result

    def test_no_domain(self):
        room = {"title": "No Domain", "content": "plain"}
        result = _format_mid(room)
        assert "# No Domain" in result
        assert "[DOMAIN:" not in result

    def test_no_tags(self):
        room = {"title": "No Tags", "domain": "test", "content": "body"}
        result = _format_mid(room)
        assert "[TOPIC:" not in result

    def test_content_as_list(self):
        room = {"title": "T", "content": ["[KEY: a] item1", "item2"]}
        result = _format_mid(room)
        assert "[KEY: a] item1" in result
        assert "item2" in result

    def test_content_as_list_of_dicts(self):
        room = {"title": "T", "content": [{"text": "[KEY: a] foo"}, {"text": "bar"}]}
        result = _format_mid(room)
        assert "[KEY: a] foo" in result
        assert "bar" in result


# ---------------------------------------------------------------------------
# _format_large (minimal structure, keys only)
# ---------------------------------------------------------------------------


class TestFormatLarge:
    def test_basic_room(self):
        result = _format_large(SAMPLE_ROOM)
        assert "Constraint Theory — Eisenstein Snap" in result
        assert "=" * len("Constraint Theory — Eisenstein Snap") in result
        assert "[KEY: Eisenstein snap]" in result
        assert "[KEY: Detection method]" in result
        # Should NOT have cross-refs, warnings, or domain tags
        assert "[CROSS-REF:" not in result
        assert "[WARNING:" not in result
        assert "[DOMAIN:" not in result

    def test_non_key_tags_stripped(self):
        result = _format_large(SAMPLE_ROOM)
        # The non-key lines should have tags stripped
        assert "Additional explanatory text for context." in result

    def test_no_content(self):
        result = _format_large({"title": "Only Title"})
        assert "Only Title" in result

    def test_uses_name_fallback(self):
        result = _format_large({"name": "Name Title", "content": "body"})
        assert "Name Title" in result
        assert "body" in result


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestDetectProfileFromName:
    def test_qwen3_0_6b_tiny(self):
        assert _detect_profile_from_name("qwen3:0.6b") == ModelProfile.TINY

    def test_qwen_0_6b_tiny(self):
        assert _detect_profile_from_name("qwen-0.6b") == ModelProfile.TINY

    def test_llama_1b_tiny(self):
        assert _detect_profile_from_name("llama3.2:1b") == ModelProfile.TINY

    def test_tinyllama_tiny(self):
        assert _detect_profile_from_name("tinyllama") == ModelProfile.TINY

    def test_glm_5_turbo_mid(self):
        assert _detect_profile_from_name("glm-5-turbo") == ModelProfile.MID

    def test_glm_4_7_mid(self):
        assert _detect_profile_from_name("glm-4.7") == ModelProfile.MID

    def test_mistral_7b_mid(self):
        assert _detect_profile_from_name("mistral-7b") == ModelProfile.MID

    def test_llama_8b_mid(self):
        assert _detect_profile_from_name("llama3.1:8b") == ModelProfile.MID

    def test_hermes_405b_large(self):
        assert _detect_profile_from_name("hermes-3-405b") == ModelProfile.LARGE

    def test_hermes_70b_large(self):
        assert _detect_profile_from_name("NousResearch/Hermes-3-Llama-3.1-70B") == ModelProfile.LARGE

    def test_seed_2_0_mini_large(self):
        assert _detect_profile_from_name("ByteDance/Seed-2.0-mini") == ModelProfile.LARGE

    def test_seed_2_0_code_large(self):
        assert _detect_profile_from_name("ByteDance/Seed-2.0-code") == ModelProfile.LARGE

    def test_deepseek_v3_large(self):
        assert _detect_profile_from_name("deepseek-v3") == ModelProfile.LARGE

    def test_deepseek_chat_large(self):
        assert _detect_profile_from_name("deepseek-chat") == ModelProfile.LARGE

    def test_claude_large(self):
        assert _detect_profile_from_name("claude-3.5-sonnet") == ModelProfile.LARGE

    def test_gpt4_large(self):
        assert _detect_profile_from_name("gpt-4") == ModelProfile.LARGE

    def test_gemini_large(self):
        assert _detect_profile_from_name("gemini-2.0-pro") == ModelProfile.LARGE

    def test_gemma_2b_mid(self):
        assert _detect_profile_from_name("gemma-2-2b") == ModelProfile.MID

    def test_qwen3_235b_large(self):
        assert _detect_profile_from_name("Qwen3-235B-A22B-Instruct-2507") == ModelProfile.LARGE

    def test_llama_70b_large(self):
        assert _detect_profile_from_name("llama-3.1-70b") == ModelProfile.LARGE

    def test_llama_90b_large(self):
        assert _detect_profile_from_name("llama-3-90b") == ModelProfile.LARGE

    def test_unknown_model_returns_none(self):
        assert _detect_profile_from_name("some-unknown-model-xyz") is None


class TestFallbackDetect:
    def test_explicit_1b(self):
        assert _fallback_detect("model-1b") == ModelProfile.TINY

    def test_explicit_7b(self):
        assert _fallback_detect("model-7b") == ModelProfile.MID

    def test_explicit_70B(self):
        assert _fallback_detect("model-70B") == ModelProfile.LARGE

    def test_no_size_hint(self):
        assert _fallback_detect("mystery-model") is None

    def test_empty_string(self):
        assert _fallback_detect("") is None


class TestAdaptiveFormatterDetect:
    def test_detect_profile_method(self):
        f = AdaptiveFormatter()
        assert f.detect_profile("qwen3:0.6b") == ModelProfile.TINY
        assert f.detect_profile("glm-5-turbo") == ModelProfile.MID
        assert f.detect_profile("hermes-3-405b") == ModelProfile.LARGE
        assert f.detect_profile("") == ModelProfile.MID  # safe default

    def test_detect_fallback_tiny_keywords(self):
        f = AdaptiveFormatter()
        assert f.detect_profile("ultra-tiny-model") == ModelProfile.TINY
        assert f.detect_profile("mini-model") == ModelProfile.TINY

    def test_detect_fallback_large_keywords(self):
        f = AdaptiveFormatter()
        assert f.detect_profile("xlarge-model") == ModelProfile.LARGE
        assert f.detect_profile("huge-network") == ModelProfile.LARGE

    def test_unknown_returns_mid(self):
        f = AdaptiveFormatter()
        assert f.detect_profile("completely-unknown-model-v99") == ModelProfile.MID


# ---------------------------------------------------------------------------
# AdaptiveFormatter.format_room
# ---------------------------------------------------------------------------


class TestFormatRoom:
    def test_tiny_profile(self):
        f = AdaptiveFormatter(ModelProfile.TINY)
        result = f.format_room(SAMPLE_ROOM, query="Eisenstein snap")
        assert "[KEY:" not in result
        assert "[CROSS-REF:" not in result
        assert "[WARNING:" not in result
        assert "[DOMAIN:" not in result
        assert "Context: Eisenstein snap" in result  # context prepended for tiny

    def test_mid_profile(self):
        f = AdaptiveFormatter(ModelProfile.MID)
        result = f.format_room(SAMPLE_ROOM, query="Eisenstein snap")
        assert "# Constraint Theory — Eisenstein Snap" in result
        assert "[DOMAIN:" in result
        assert "[KEY:" in result
        assert "[CROSS-REF:" in result

    def test_large_profile(self):
        f = AdaptiveFormatter(ModelProfile.LARGE)
        result = f.format_room(SAMPLE_ROOM)
        assert "[KEY: Eisenstein snap]" in result
        assert "[CROSS-REF:" not in result
        assert "[WARNING:" not in result
        assert "[DOMAIN:" not in result
        assert "=" * len("Constraint Theory — Eisenstein Snap") in result

    def test_auto_with_model_name_tiny(self):
        f = AdaptiveFormatter()
        result = f.format_room(SAMPLE_ROOM, model_name="qwen3:0.6b")
        assert "[KEY:" not in result

    def test_auto_with_model_name_mid(self):
        f = AdaptiveFormatter()
        result = f.format_room(SAMPLE_ROOM, model_name="glm-5-turbo")
        assert "[KEY:" in result

    def test_auto_with_model_name_large(self):
        f = AdaptiveFormatter()
        result = f.format_room(SAMPLE_ROOM, model_name="seed-2.0-mini")
        assert "[KEY: Eisenstein snap]" in result
        assert "[CROSS-REF:" not in result

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError):
            AdaptiveFormatter("invalid_profile")

    def test_empty_room(self):
        f = AdaptiveFormatter(ModelProfile.TINY)
        result = f.format_room({})
        assert result == ""

    def test_room_none_title(self):
        f = AdaptiveFormatter(ModelProfile.MID)
        result = f.format_room({"content": "body"})
        assert "body" in result


# ---------------------------------------------------------------------------
# AdaptiveFormatter.format_rooms
# ---------------------------------------------------------------------------


class TestFormatRooms:
    def test_multiple_rooms_tiny(self):
        f = AdaptiveFormatter(ModelProfile.TINY)
        rooms = [
            {"title": "Room A", "content": "Content A", "domain": "physics"},
            {"title": "Room B", "content": "Content B", "domain": "math"},
        ]
        result = f.format_rooms(rooms, query="physics")
        # Tiny: no headings, no tags, no scores
        assert "[KEY:" not in result
        assert "score=" not in result

    def test_multiple_rooms_mid(self):
        f = AdaptiveFormatter(ModelProfile.MID)
        rooms = [
            {"title": "Room A", "content": "physics content", "domain": "physics"},
            {"title": "Room B", "content": "math content", "domain": "math"},
        ]
        result = f.format_rooms(rooms, query="physics")
        # Room A (physics) should be first — score lines present
        assert "Room 1" in result
        assert "score=" in result

    def test_unknown_rooms_skipped(self):
        f = AdaptiveFormatter()
        rooms: list[dict] = [{"title": "Valid", "content": "ok"}, "not a dict"]
        result = f.format_rooms(rooms, query="")
        assert "Valid" in result or "ok" in result

    def test_empty_rooms_list(self):
        f = AdaptiveFormatter()
        result = f.format_rooms([], query="anything")
        assert result == ""

    def test_all_irrelevant_with_query(self):
        f = AdaptiveFormatter()
        rooms = [{"title": "One", "content": "aaa"}, {"title": "Two", "content": "bbb"}]
        result = f.format_rooms(rooms, query="zzzzz_nonexistent_zzzzz")
        # No rooms should match
        assert result == ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestComputeRelevance:
    def test_exact_phrase_match(self):
        room = {"title": "The Thing", "content": "Eisenstein snap detection is important."}
        score = compute_relevance("Eisenstein snap", room)
        assert score >= 10.0  # exact phrase bonus

    def test_no_match(self):
        room = {"title": "Unrelated", "content": "nothing in common"}
        score = compute_relevance("Eisenstein", room)
        assert score == 0.0

    def test_domain_bonus(self):
        room = {"title": "T", "content": "content", "domain": "constraint-theory"}
        score = compute_relevance("constraint", room)
        assert score >= 5.0  # domain match bonus

    def test_tag_bonus(self):
        room = {"title": "T", "content": "content", "tags": ["eisenstein"]}
        score = compute_relevance("Eisenstein", room)
        assert score >= 3.0  # tag match bonus

    def test_empty_query(self):
        assert compute_relevance("", SAMPLE_ROOM) == 0.0


class TestComputeRecency:
    def test_recent_room_scores_high(self):
        room = {"updated": "2026-05-13T00:00:00Z"}
        from datetime import datetime, timezone, timedelta
        now = datetime(2026, 5, 13, tzinfo=timezone.utc)
        score = compute_recency(room, now=now)
        assert score > 4.0

    def test_old_room_scores_low(self):
        room = {"updated": "2025-01-01T00:00:00Z"}
        from datetime import datetime, timezone
        now = datetime(2026, 5, 13, tzinfo=timezone.utc)
        score = compute_recency(room, now=now)
        assert score < 3.0

    def test_no_date_returns_zero(self):
        assert compute_recency({}) == 0.0

    def test_created_field_as_fallback(self):
        room = {"created": "2026-05-01T12:00:00Z"}
        score = compute_recency(room)
        assert isinstance(score, float)

    def test_date_field_as_fallback(self):
        room = {"date": "2026-05-10"}
        score = compute_recency(room)
        assert isinstance(score, float)


class TestComputeDomainDensity:
    def test_full_room(self):
        room = {
            "domain": "test",
            "tags": ["a", "b"],
            "content": "[KEY: foo] bar",
        }
        assert compute_domain_density(room) == 3.0

    def test_minimal_room(self):
        assert compute_domain_density({"title": "Just Title"}) == 0.0

    def test_partial(self):
        room = {"domain": "test", "content": "[KEY: foo] bar"}
        assert compute_domain_density(room) == 2.0


class TestScoreRoom:
    def test_combined_score(self):
        room = {
            "title": "Eisenstein Snap Detection",
            "content": "Eisenstein snap detection method and analysis.",
            "domain": "constraint-theory",
            "tags": ["eisenstein"],
            "updated": "2026-05-12T14:30:00Z",
        }
        s = score_room("Eisenstein snap", room)
        assert s > 0

    def test_irrelevant_room_scores_low(self):
        room = {
            "title": "Unrelated",
            "content": "nothing in common here at all",
        }
        s = score_room("Eisenstein snap", room)
        # Only recency might contribute, but no date = 0
        assert s == 0.0


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


class TestCLI:
    def test_detect_mode(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "adaptive_plato", "--detect", "qwen3:0.6b"],
            capture_output=True, text=True, cwd=HERE
        )
        assert result.returncode == 0
        assert "qwen3:0.6b" in result.stdout
        assert "tiny" in result.stdout

    def test_detect_large(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "adaptive_plato", "--detect", "seed-2.0-pro"],
            capture_output=True, text=True, cwd=HERE
        )
        assert result.returncode == 0
        assert "large" in result.stdout

    def test_format_with_model(self):
        import subprocess
        rooms_path = HERE / "rooms.json"
        result = subprocess.run(
            [
                "python3", "-m", "adaptive_plato",
                str(rooms_path),
                "--model", "glm-5-turbo",
                "--query", "Eisenstein snap",
            ],
            capture_output=True, text=True, cwd=HERE
        )
        assert result.returncode == 0
        assert "Eisenstein" in result.stdout or "snap" in result.stdout or "Room 1" in result.stdout

    def test_format_tiny_profile(self):
        import subprocess
        rooms_path = HERE / "rooms.json"
        result = subprocess.run(
            [
                "python3", "-m", "adaptive_plato",
                str(rooms_path),
                "--profile", "tiny",
                "--query", "Eisenstein snap",
            ],
            capture_output=True, text=True, cwd=HERE
        )
        assert result.returncode == 0
        # Tiny should NOT have tags
        assert "[KEY:" not in result.stdout
        assert "[CROSS-REF:" not in result.stdout


HERE = Path(__file__).parent

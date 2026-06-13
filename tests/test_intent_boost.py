"""Unit tests for the declarative intent-boost helpers extracted from node_prefilter_tools (W3.4)."""

from universal_agentic_framework.orchestration.helpers.tool_scoring import (
    intent_boost_applies,
    apply_intent_override_floor,
)


def _intents(**overrides):
    """A zeroed intents dict with only the given keys turned on."""
    base = {
        "mentions_datetime": False, "mentions_calculation": False, "url_in_query": None,
        "mentions_web_search": False, "image_url_in_query": False, "image_in_query": False,
        "mentions_ocr": False, "mentions_document": False, "mentions_chart": False,
        "mentions_image_metadata": False, "mentions_barcode": False, "mentions_map": False,
        "mentions_csv_analysis": False,
    }
    base.update(overrides)
    return base


def _applies(tool, intents, *, img=False, csv=False):
    return intent_boost_applies(tool, intents, image_attachment_present=img, csv_workspace_doc_present=csv)


# ── intent_boost_applies ──────────────────────────────────────────────

def test_simple_intent_match():
    assert _applies("datetime_tool", _intents(mentions_datetime=True))
    assert not _applies("datetime_tool", _intents(mentions_datetime=False))
    assert _applies("calculator_tool", _intents(mentions_calculation=True))
    assert _applies("web_search_mcp", _intents(mentions_web_search=True))
    assert _applies("extract_webpage_mcp", _intents(url_in_query="https://x.com"))


def test_vision_tool_requires_image_and_keyword():
    # ocr_tool needs an image (url or attachment) AND the OCR keyword.
    assert not _applies("ocr_tool", _intents(mentions_ocr=True))  # no image
    assert not _applies("ocr_tool", _intents(image_in_query=True))  # no keyword
    assert _applies("ocr_tool", _intents(image_in_query=True, mentions_ocr=True))
    assert _applies("ocr_tool", _intents(mentions_ocr=True), img=True)  # attachment satisfies image


def test_analyze_image_matches_on_attachment_alone():
    assert _applies("analyze_image_tool", _intents(), img=True)
    assert _applies("analyze_image_tool", _intents(image_url_in_query=True))
    assert not _applies("analyze_image_tool", _intents())


def test_csv_requires_intent_and_doc_present():
    assert not _applies("csv_analyze_tool", _intents(mentions_csv_analysis=True))  # no doc
    assert not _applies("csv_analyze_tool", _intents(), csv=True)  # no intent
    assert _applies("csv_analyze_tool", _intents(mentions_csv_analysis=True), csv=True)


def test_unknown_tool_never_boosts():
    assert not _applies("some_random_tool", _intents(mentions_datetime=True))


# ── apply_intent_override_floor ───────────────────────────────────────

def test_override_floor_lifts_low_score():
    sim, applied = apply_intent_override_floor(
        "web_search_mcp", 0.30, _intents(mentions_web_search=True),
        similarity_threshold=0.55, min_top_score=0.7,
    )
    assert applied is True
    assert sim == 0.71  # max(0.55, 0.7) + 0.01


def test_override_floor_noop_when_already_high():
    sim, applied = apply_intent_override_floor(
        "web_search_mcp", 0.95, _intents(mentions_web_search=True),
        similarity_threshold=0.55, min_top_score=0.7,
    )
    assert applied is False
    assert sim == 0.95


def test_override_floor_requires_intent():
    sim, applied = apply_intent_override_floor(
        "web_search_mcp", 0.30, _intents(mentions_web_search=False),
        similarity_threshold=0.55, min_top_score=0.7,
    )
    assert applied is False
    assert sim == 0.30


def test_override_floor_only_for_eligible_tools():
    sim, applied = apply_intent_override_floor(
        "datetime_tool", 0.30, _intents(mentions_datetime=True),
        similarity_threshold=0.55, min_top_score=0.7,
    )
    assert applied is False
    assert sim == 0.30

    sim_map, applied_map = apply_intent_override_floor(
        "map_tool", 0.10, _intents(mentions_map=True),
        similarity_threshold=0.55, min_top_score=0.7,
    )
    assert applied_map is True
    assert sim_map == 0.71

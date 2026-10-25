"""Tests for text processing helpers (truncation, context blocks)."""

import pytest

from universal_agentic_framework.orchestration.helpers.text_processing import (
    truncate_tabular_by_rows,
    truncate_text_by_tokens,
    build_workspace_document_context_block,
    build_workspace_tool_paths,
)


# ── truncate_tabular_by_rows ───────────────────────────────────────────────────


class TestTruncateTabularByRows:
    def test_fits_within_budget_returns_unchanged(self):
        text = "a,b,c\n1,2,3\n4,5,6\n"
        result = truncate_tabular_by_rows(text, max_tokens=200)
        assert "1,2,3" in result
        assert "4,5,6" in result
        assert "omitted" not in result

    def test_always_keeps_header(self):
        header = "col1,col2,col3"
        rows = "\n".join(f"val{i},val{i},val{i}" for i in range(50))
        text = header + "\n" + rows
        result = truncate_tabular_by_rows(text, max_tokens=10)
        assert result.startswith("col1,col2,col3")

    def test_never_cuts_mid_row(self):
        header = "name,score"
        rows = "\n".join(f"user{i},{i*10}" for i in range(100))
        text = header + "\n" + rows
        result = truncate_tabular_by_rows(text, max_tokens=30)
        lines = result.split("\n")
        for line in lines:
            if line.startswith("[…"):
                continue
            assert "," in line or line == header

    def test_truncation_note_included(self):
        header = "x"
        rows = "\n".join(str(i) for i in range(100))
        text = header + "\n" + rows
        result = truncate_tabular_by_rows(text, max_tokens=20)
        assert "omitted" in result
        assert "of 100" in result

    def test_empty_text_returns_empty(self):
        assert truncate_tabular_by_rows("", max_tokens=100) == ""

    def test_zero_max_tokens_returns_empty(self):
        assert truncate_tabular_by_rows("a,b\n1,2\n", max_tokens=0) == ""

    def test_note_counts_data_rows_not_header(self):
        header = "description,amount,category,date,status"
        rows = "\n".join(f"long_description_item_{i},amount_{i},category_{i},2024-01-{i:02d},active" for i in range(1, 21))
        text = header + "\n" + rows
        result = truncate_tabular_by_rows(text, max_tokens=30)
        # Must reference 20 total data rows, not 21
        if "omitted" in result:
            assert "of 20 omitted" in result


# ── build_workspace_document_context_block ────────────────────────────────────


class TestBuildWorkspaceDocumentContextBlock:
    def _make_doc(self, filename, content, mime_type="text/plain", stored_path="/tmp/ws/f.txt"):
        return {
            "id": "doc1",
            "filename": filename,
            "version": 1,
            "mime_type": mime_type,
            "size_bytes": len(content),
            "content_text": content,
            "stored_path": stored_path,
        }

    def test_non_csv_unchanged_behaviour(self):
        doc = self._make_doc("notes.md", "Hello world", mime_type="text/markdown")
        block, ctx = build_workspace_document_context_block([doc])
        assert "Hello world" in block
        assert len(ctx) == 1
        # No CSV path hint
        assert "csv_analyze_tool" not in block

    def test_csv_uses_larger_budget(self):
        # A large CSV fills the budget with whole rows; a short note stays short.
        header = "col1,col2"
        rows = "\n".join(f"value{i},value{i}" for i in range(500))
        csv_content = header + "\n" + rows
        doc = self._make_doc(
            "data.csv", csv_content, mime_type="text/csv",
            stored_path="/tmp/ws/data.csv",
        )
        md_doc = self._make_doc("notes.md", "short note", mime_type="text/markdown")

        block_csv, _ = build_workspace_document_context_block([doc])
        block_md, _ = build_workspace_document_context_block([md_doc])

        # CSV block should contain more rows than the plain-text budget would allow
        assert len(block_csv) > len(block_md)

    def test_csv_path_hint_present(self):
        doc = self._make_doc(
            "budget.csv", "item,amount\nfoo,100\nbar,200\n",
            mime_type="text/csv", stored_path="/tmp/ws/budget.csv",
        )
        block, _ = build_workspace_document_context_block([doc])
        assert "csv_analyze_tool" in block
        assert "/tmp/ws/budget.csv" in block

    def test_csv_detected_by_extension_when_no_mime(self):
        doc = self._make_doc(
            "report.csv", "a,b\n1,2\n",
            mime_type="text/plain", stored_path="/tmp/ws/report.csv",
        )
        block, _ = build_workspace_document_context_block([doc])
        assert "csv_analyze_tool" in block

    def test_non_csv_has_no_path_hint(self):
        doc = self._make_doc("doc.txt", "some text", mime_type="text/plain")
        block, _ = build_workspace_document_context_block([doc])
        assert "csv_analyze_tool" not in block

    def test_small_doc_kept_whole_beside_large_csv(self):
        # Budget is split by need (water-filling): the short text doc gets all it
        # needs and the CSV takes the rest, instead of the CSV starving it.
        csv_doc = self._make_doc(
            "data.csv", "x,y\n" + "\n".join(f"{i},{i}" for i in range(10)),
            mime_type="text/csv", stored_path="/tmp/ws/d.csv",
        )
        text_doc = self._make_doc("notes.md", "Important notes here.", mime_type="text/markdown")
        block, ctx = build_workspace_document_context_block([csv_doc, text_doc])
        assert "Important notes here." in block

    def test_empty_documents_returns_empty(self):
        block, ctx = build_workspace_document_context_block([])
        assert block == ""
        assert ctx == []

    def test_row_aware_truncation_no_mid_row_cut(self):
        header = "description,amount,category,subcategory,date,status,notes"
        rows = "\n".join(
            f"very_long_description_item_{i},amount_value_{i},main_category_{i},sub_category_{i},2024-01-{(i%28)+1:02d},active_status,some_notes_{i}"
            for i in range(500)
        )
        doc = self._make_doc(
            "big.csv", header + "\n" + rows,
            mime_type="text/csv", stored_path="/tmp/ws/big.csv",
        )
        block, _ = build_workspace_document_context_block([doc])
        # 500 long rows exceed the default budget: the excerpt names the omitted row
        # range rather than cutting mid-row, and keeps the newest (last) row.
        assert "EXCERPT" in block
        assert "of 500 omitted" in block
        assert "some_notes_499" in block


# ── build_workspace_tool_paths ─────────────────────────────────────────────────


class TestBuildWorkspaceToolPaths:
    def test_csv_doc_produces_path_section(self):
        docs = [{"filename": "sales.csv", "mime_type": "text/csv", "stored_path": "/tmp/ws/sales.csv"}]
        result = build_workspace_tool_paths(docs)
        assert "sales.csv" in result
        assert "/tmp/ws/sales.csv" in result
        assert "csv_analyze_tool" in result

    def test_no_csv_docs_returns_empty(self):
        docs = [{"filename": "notes.md", "mime_type": "text/markdown", "stored_path": "/tmp/ws/n.md"}]
        assert build_workspace_tool_paths(docs) == ""

    def test_empty_list_returns_empty(self):
        assert build_workspace_tool_paths([]) == ""

    def test_doc_without_stored_path_skipped(self):
        docs = [{"filename": "x.csv", "mime_type": "text/csv", "stored_path": ""}]
        assert build_workspace_tool_paths(docs) == ""

    def test_csv_detected_by_extension(self):
        docs = [{"filename": "data.csv", "mime_type": "text/plain", "stored_path": "/tmp/ws/data.csv"}]
        result = build_workspace_tool_paths(docs)
        assert "/tmp/ws/data.csv" in result

    def test_multiple_csv_docs(self):
        docs = [
            {"filename": "a.csv", "mime_type": "text/csv", "stored_path": "/tmp/ws/a.csv"},
            {"filename": "b.csv", "mime_type": "text/csv", "stored_path": "/tmp/ws/b.csv"},
        ]
        result = build_workspace_tool_paths(docs)
        assert "a.csv" in result
        assert "b.csv" in result


# ── Intent detection for CSV analysis ─────────────────────────────────────────


class TestCsvAnalysisIntentDetection:
    def _intents(self, msg, lang="en"):
        from universal_agentic_framework.orchestration.helpers.intent_detection import (
            detect_tool_routing_intents,
        )
        return detect_tool_routing_intents(msg, lang)

    def test_sum_triggers(self):
        assert self._intents("sum the amount column")["mentions_csv_analysis"]

    def test_average_triggers(self):
        assert self._intents("what is the average price?")["mentions_csv_analysis"]

    def test_count_rows_triggers(self):
        assert self._intents("how many rows are there?")["mentions_csv_analysis"]

    def test_group_by_triggers(self):
        assert self._intents("group by category")["mentions_csv_analysis"]

    def test_filter_triggers(self):
        assert self._intents("filter where status == active")["mentions_csv_analysis"]

    def test_de_summe_triggers(self):
        assert self._intents("Berechne die Summe der Spalte", lang="de")["mentions_csv_analysis"]

    def test_de_durchschnitt_triggers(self):
        assert self._intents("Was ist der Durchschnitt?", lang="de")["mentions_csv_analysis"]

    def test_de_wie_viele_zeilen(self):
        assert self._intents("Wie viele Zeilen hat die Tabelle?", lang="de")["mentions_csv_analysis"]

    def test_de_csv_auswerten(self):
        assert self._intents("CSV auswerten", lang="de")["mentions_csv_analysis"]

    def test_general_question_does_not_trigger(self):
        assert not self._intents("what is the capital of France?")["mentions_csv_analysis"]

    def test_mentions_csv_analysis_in_returned_dict(self):
        result = self._intents("hello")
        assert "mentions_csv_analysis" in result


def test_latest_user_message_skips_trailing_assistant_output():
    from universal_agentic_framework.orchestration.helpers.text_processing import latest_user_message

    messages = [
        {"role": "user", "content": "compare solar vs wind"},
        {"role": "assistant", "content": "Solar is ..."},
    ]
    assert latest_user_message(messages) == "compare solar vs wind"
    assert latest_user_message([]) == ""
    assert latest_user_message(None) == ""


# ── Layer 3: local-path tool args are confined to this turn's own files ───────


def test_local_path_arg_must_be_a_turn_file():
    import pytest as _pytest

    from universal_agentic_framework.orchestration.helpers import (
        check_local_path_args,
        turn_local_paths,
    )

    allowed = turn_local_paths(
        [{"stored_path": "/ws/conv-a/attachments/cat.png"}],
        [{"stored_path": "/ws/u1/docs/sales.csv"}, {"stored_path": None}],
    )
    check_local_path_args({"image_source": "/ws/conv-a/attachments/cat.png"}, allowed)
    check_local_path_args({"file_path": "/ws/u1/docs/./sales.csv"}, allowed)  # normalized
    check_local_path_args({"image_source": "https://cdn.example.com/x.png"}, allowed)
    check_local_path_args({"query": "unrelated"}, allowed)

    with _pytest.raises(ValueError, match="attached to this conversation"):
        check_local_path_args({"image_source": "/ws/conv-other-user/attachments/id.png"}, allowed)
    with _pytest.raises(ValueError):
        check_local_path_args({"file_path": "/ws/u1/docs/../../u2/docs/payroll.csv"}, allowed)


# ── Content-aware estimate, excerpts, budgets ─────────────────────────────────

import json as _json

from universal_agentic_framework.llm.budget import estimate_content_tokens, estimate_tokens
from universal_agentic_framework.orchestration.helpers.text_processing import (
    allocate_token_budgets,
    build_attachment_context_block,
    build_context_manifest,
)


def _consumption_export(days: int = 676) -> str:
    """Minified JSON shaped like a real export: records under "data" plus a "meta" tail."""
    import datetime as _dt

    start = _dt.date(2024, 12, 11)
    data = [
        {"date": (start + _dt.timedelta(days=i)).isoformat(), "grams": round(40 + (i * 7.3) % 70, 1)}
        for i in range(days)
    ]
    return _json.dumps({"data": data, "meta": {"exported_at": "2026-10-18T15:03:39Z", "user_id": 1}},
                       separators=(",", ":"))


class TestEstimateContentTokens:
    def test_numeric_json_counts_far_above_chars_over_4(self):
        text = _consumption_export(200)
        # Real Gemma count for this shape is ~len/1.75; chars/4 undercounts it ~2.3x.
        assert estimate_content_tokens(text) > 2 * estimate_tokens(text)
        assert estimate_content_tokens(text) >= len(text) / 1.9

    def test_prose_stays_close_to_chars_over_4(self):
        text = "The committee discusses budget allocations for the upcoming fiscal year. " * 30
        assert 0.8 * estimate_tokens(text) <= estimate_content_tokens(text) <= 1.5 * estimate_tokens(text)

    def test_empty(self):
        assert estimate_content_tokens("") == 0


class TestExcerptText:
    def test_keeps_head_and_tail_with_explicit_marker(self):
        text = "\n".join(f"line {i}: entry" for i in range(2000))
        out = truncate_text_by_tokens(text, max_tokens=300)
        assert out.startswith("line 0: entry")
        assert out.endswith("line 1999: entry")
        assert "NOT visible to you" in out
        assert estimate_content_tokens(out) <= 300 + 40

    def test_fitting_text_returned_unchanged(self):
        assert truncate_text_by_tokens("short note", max_tokens=100) == "short note"


class TestJsonExcerpt:
    def _doc(self, content, filename="alcohol-consumption.json", mime="application/json"):
        return {
            "id": "d1", "filename": filename, "version": 1, "mime_type": mime,
            "size_bytes": len(content), "content_text": content, "stored_path": "/tmp/ws/a.json",
        }

    def test_excerpt_is_valid_json_with_newest_record_and_meta(self):
        content = _consumption_export()
        block, ctx = build_workspace_document_context_block([self._doc(content)], total_budget_tokens=4000)
        snippet = ctx[0]["text"]
        parsed = _json.loads(snippet)  # still valid JSON
        assert parsed["meta"]["user_id"] == 1
        assert parsed["data"][-1] == _json.loads(content)["data"][-1]  # the latest entry survives
        assert parsed["data"][0] == _json.loads(content)["data"][0]
        assert "items" in ctx[0]["excerpt"] and 'in "data"' in ctx[0]["excerpt"]
        assert "| EXCERPT — items" in block
        assert "[Full file at: /tmp/ws/a.json" in block and "records" in block
        assert estimate_content_tokens(block) <= 4000 + 200

    def test_top_level_array(self):
        content = _json.dumps([{"i": i, "v": "x" * 20} for i in range(500)])
        _, ctx = build_workspace_document_context_block([self._doc(content)], total_budget_tokens=1000)
        parsed = _json.loads(ctx[0]["text"])
        assert parsed[0]["i"] == 0 and parsed[-1]["i"] == 499

    def test_invalid_json_falls_back_to_head_and_tail(self):
        content = '{"data":[' + ",".join(f'{{"n":{i}}}' for i in range(3000))  # truncated, no closing
        _, ctx = build_workspace_document_context_block([self._doc(content)], total_budget_tokens=800)
        assert ctx[0]["excerpt"].startswith("characters")
        assert ctx[0]["text"].rstrip().endswith('{"n":2999}')

    def test_complete_doc_has_no_excerpt_label(self):
        content = _consumption_export(10)
        block, ctx = build_workspace_document_context_block([self._doc(content)])
        assert ctx[0]["excerpt"] is None
        assert "EXCERPT" not in block
        assert ctx[0]["text"] == content


class TestCsvHeadAndTail:
    def test_csv_excerpt_keeps_last_rows(self):
        header = "date,grams"
        rows = [f"2026-01-{(i % 28) + 1:02d},{i}" for i in range(1000)]
        out = truncate_tabular_by_rows(header + "\n" + "\n".join(rows), max_tokens=300)
        lines = out.split("\n")
        assert lines[0] == header
        assert lines[1] == rows[0]
        assert lines[-1] == rows[-1]
        assert any("data rows" in line and "of 1,000" in line for line in lines)


class TestAllocateTokenBudgets:
    def test_small_items_get_their_need_large_split_rest(self):
        assert allocate_token_budgets([100, 5000, 5000], 3000) == [100, 1450, 1450]

    def test_everything_fits(self):
        assert allocate_token_budgets([100, 200], 3000) == [100, 200]

    def test_zero_need_item(self):
        assert allocate_token_budgets([0, 9000], 4000) == [0, 4000]


class TestContextManifest:
    def test_lists_complete_and_excerpted_files(self):
        _, att_ctx = build_attachment_context_block(
            [{"id": "a", "original_name": "notes.md", "mime_type": "text/markdown", "extracted_text": "UNIQUE-BODY-7731"}]
        )
        ws_ctx = [{"filename": "export.json", "version": 2, "excerpt": "items 5–90 of 100 omitted"}]
        manifest = build_context_manifest(att_ctx, ws_ctx)
        assert "- Attachment notes.md: complete" in manifest
        assert "- Workspace document export.json v2: EXCERPT (items 5–90 of 100 omitted)" in manifest
        assert "UNIQUE-BODY-7731" not in manifest  # names only, never content

    def test_empty_when_no_files(self):
        assert build_context_manifest([], []) == ""


class TestDataFileToolPaths:
    def test_json_doc_listed_for_csv_analyze(self):
        docs = [{"filename": "export.json", "mime_type": "application/json", "stored_path": "/tmp/ws/e.json"}]
        result = build_workspace_tool_paths(docs)
        assert "[JSON data: export.json]" in result
        assert "/tmp/ws/e.json" in result
        assert "csv_analyze_tool" in result

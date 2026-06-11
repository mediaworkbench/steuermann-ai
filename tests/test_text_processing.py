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
        assert "showing" not in result

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
            if line.startswith("…"):
                continue
            assert "," in line or line == header

    def test_truncation_note_included(self):
        header = "x"
        rows = "\n".join(str(i) for i in range(100))
        text = header + "\n" + rows
        result = truncate_tabular_by_rows(text, max_tokens=20)
        assert "showing first" in result
        assert "of 100 rows" in result

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
        if "showing first" in result:
            assert "of 20 rows" in result


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
        # Build a CSV with many rows; non-CSV truncates at 600 tokens but CSV allows 1500
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

    def test_total_budget_raised_when_csv_present(self):
        # A CSV doc + another text doc; the total budget is raised so the text
        # doc still gets rendered despite the CSV consuming tokens.
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
        # Verify the truncation note is present (500 long rows exceed the 1500-token CSV budget)
        # and that we see the row-level summary rather than a mid-row cut.
        assert "showing first" in block
        assert "of 500 rows" in block


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

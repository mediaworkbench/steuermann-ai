"""Tests for csv_analyze_tool."""

import os
from pathlib import Path

import pytest

from universal_agentic_framework.tools.csv_analyze.tool import (
    CsvAnalyzeInput,
    CsvAnalyzeTool,
    _coerce_numeric,
    _sniff_delimiter,
    _find_column,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


class TestCoerceNumeric:
    def test_plain_integer(self):
        assert _coerce_numeric("42") == 42.0

    def test_plain_float(self):
        assert _coerce_numeric("3.14") == pytest.approx(3.14)

    def test_en_thousands(self):
        assert _coerce_numeric("1,234.56") == pytest.approx(1234.56)

    def test_de_thousands(self):
        assert _coerce_numeric("1.234,56") == pytest.approx(1234.56)

    def test_de_comma_decimal_no_thousands(self):
        assert _coerce_numeric("99,50") == pytest.approx(99.50)

    def test_euro_prefix(self):
        assert _coerce_numeric("€ 1.234,56") == pytest.approx(1234.56)

    def test_dollar_prefix(self):
        assert _coerce_numeric("$1,000.00") == pytest.approx(1000.0)

    def test_negative(self):
        assert _coerce_numeric("-42.5") == pytest.approx(-42.5)

    def test_non_numeric(self):
        assert _coerce_numeric("hello") is None

    def test_empty(self):
        assert _coerce_numeric("") is None


class TestSniffDelimiter:
    def test_comma(self):
        sample = "name,age,city\nAlice,30,Berlin"
        assert _sniff_delimiter(sample) == ","

    def test_semicolon_de(self):
        sample = "name;age;city\nAlice;30;Berlin"
        assert _sniff_delimiter(sample) == ";"

    def test_tab(self):
        sample = "name\tage\tcity\nAlice\t30\tBerlin"
        assert _sniff_delimiter(sample) == "\t"


class TestFindColumn:
    def test_exact_match(self):
        assert _find_column(["name", "age"], "age") == "age"

    def test_case_insensitive(self):
        assert _find_column(["Name", "Age"], "name") == "Name"

    def test_not_found(self):
        assert _find_column(["name", "age"], "city") is None

    def test_strip_whitespace(self):
        assert _find_column([" amount "], " amount ") == " amount "


# ── Tool integration ───────────────────────────────────────────────────────────


@pytest.fixture
def csv_dir(tmp_path):
    return tmp_path


@pytest.fixture
def tool(csv_dir):
    t = CsvAnalyzeTool()
    t.workspace_base_dir = str(csv_dir)
    return t


def _write_csv(directory: Path, name: str, content: str) -> str:
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestSummaryOperation:
    def test_basic_summary(self, tool, csv_dir):
        path = _write_csv(csv_dir, "data.csv", "name,amount\nAlice,100\nBob,200\nCarol,300\n")
        result = tool._run(file_path=path, operation="summary")
        assert "Rows: 3" in result
        assert "Columns (2)" in result
        assert "name" in result
        assert "amount" in result

    def test_numeric_column_detected(self, tool, csv_dir):
        path = _write_csv(csv_dir, "nums.csv", "x,y\n1,a\n2,b\n3,c\n")
        result = tool._run(file_path=path, operation="summary")
        assert "numeric" in result.lower()

    def test_empty_csv_no_crash(self, tool, csv_dir):
        path = _write_csv(csv_dir, "empty.csv", "col1,col2\n")
        result = tool._run(file_path=path, operation="summary")
        assert "Rows: 0" in result


class TestAggregateOperation:
    def test_sum(self, tool, csv_dir):
        path = _write_csv(csv_dir, "agg.csv", "item,price\nA,10\nB,20\nC,30\n")
        result = tool._run(file_path=path, operation="aggregate", column="price", aggregation="sum")
        assert "60" in result

    def test_mean(self, tool, csv_dir):
        path = _write_csv(csv_dir, "mean.csv", "x\n10\n20\n30\n")
        result = tool._run(file_path=path, operation="aggregate", column="x", aggregation="mean")
        assert "20" in result

    def test_min_max(self, tool, csv_dir):
        path = _write_csv(csv_dir, "minmax.csv", "v\n5\n1\n9\n3\n")
        assert "1" in tool._run(file_path=path, operation="aggregate", column="v", aggregation="min")
        assert "9" in tool._run(file_path=path, operation="aggregate", column="v", aggregation="max")

    def test_count(self, tool, csv_dir):
        path = _write_csv(csv_dir, "cnt.csv", "v\n1\n2\n\n4\n")
        result = tool._run(file_path=path, operation="aggregate", column="v", aggregation="count")
        assert "3" in result

    def test_group_by(self, tool, csv_dir):
        path = _write_csv(csv_dir, "grp.csv", "cat,val\nA,10\nB,5\nA,20\nB,15\n")
        result = tool._run(file_path=path, operation="aggregate", column="val", aggregation="sum", group_by="cat")
        assert "30" in result  # A sum
        assert "20" in result  # B sum

    def test_missing_column_error(self, tool, csv_dir):
        path = _write_csv(csv_dir, "err.csv", "x\n1\n")
        result = tool._run(file_path=path, operation="aggregate", column="nonexistent", aggregation="sum")
        assert result.startswith("Error:")

    def test_no_numeric_values(self, tool, csv_dir):
        path = _write_csv(csv_dir, "text.csv", "name\nAlice\nBob\n")
        result = tool._run(file_path=path, operation="aggregate", column="name", aggregation="sum")
        assert "No numeric values" in result

    def test_de_formatted_numbers(self, tool, csv_dir):
        path = _write_csv(csv_dir, "de.csv", "betrag\n1.234,56\n2.000,00\n")
        result = tool._run(file_path=path, operation="aggregate", column="betrag", aggregation="sum")
        assert "3234" in result.replace(",", "").replace(".", "")


class TestFilterOperation:
    def test_equals_string(self, tool, csv_dir):
        path = _write_csv(csv_dir, "fil.csv", "name,age\nAlice,30\nBob,25\nAlice,35\n")
        result = tool._run(file_path=path, operation="filter", column="name", filter_op="==", filter_value="Alice")
        assert "Alice" in result
        assert "2 matching rows" in result

    def test_greater_than(self, tool, csv_dir):
        path = _write_csv(csv_dir, "gt.csv", "score\n10\n80\n50\n90\n")
        result = tool._run(file_path=path, operation="filter", column="score", filter_op=">", filter_value="60")
        assert "2 matching rows" in result

    def test_contains(self, tool, csv_dir):
        path = _write_csv(csv_dir, "cont.csv", "desc\nfoo bar\nbaz\nfoo qux\n")
        result = tool._run(file_path=path, operation="filter", column="desc", filter_op="contains", filter_value="foo")
        assert "2 matching rows" in result

    def test_limit_respected(self, tool, csv_dir):
        rows = "\n".join(f"row{i},{i}" for i in range(100))
        path = _write_csv(csv_dir, "big.csv", f"name,val\n{rows}\n")
        result = tool._run(file_path=path, operation="filter", column="val", filter_op=">", filter_value="-1", limit=5)
        assert "showing up to 5" in result


class TestHeadTailOperation:
    def test_head(self, tool, csv_dir):
        path = _write_csv(csv_dir, "ht.csv", "x\n1\n2\n3\n4\n5\n")
        result = tool._run(file_path=path, operation="head", limit=3)
        assert "1" in result
        assert "4" not in result

    def test_tail(self, tool, csv_dir):
        path = _write_csv(csv_dir, "ht2.csv", "x\n1\n2\n3\n4\n5\n")
        result = tool._run(file_path=path, operation="tail", limit=2)
        assert "5" in result
        assert "1" not in result


class TestUniqueValueCounts:
    def test_unique(self, tool, csv_dir):
        path = _write_csv(csv_dir, "uniq.csv", "cat\nA\nB\nA\nC\nB\nA\n")
        result = tool._run(file_path=path, operation="unique", column="cat")
        assert "3 distinct" in result

    def test_value_counts(self, tool, csv_dir):
        path = _write_csv(csv_dir, "vc.csv", "cat\nA\nB\nA\nC\nB\nA\n")
        result = tool._run(file_path=path, operation="value_counts", column="cat")
        assert "'A': 3" in result


class TestDelimiterDetection:
    def test_semicolon_de_csv(self, tool, csv_dir):
        path = _write_csv(csv_dir, "de.csv", "Name;Betrag;Stadt\nAlice;1234;Berlin\nBob;5678;München\n")
        result = tool._run(file_path=path, operation="aggregate", column="Betrag", aggregation="sum")
        assert "6912" in result

    def test_quoted_field_with_embedded_comma(self, tool, csv_dir):
        content = 'name,description\nAlice,"hello, world"\nBob,plain\n'
        path = _write_csv(csv_dir, "quoted.csv", content)
        result = tool._run(file_path=path, operation="summary")
        assert "Rows: 2" in result


class TestPathConfinement:
    def test_path_outside_base_dir_rejected(self, tool, tmp_path):
        outside = tmp_path.parent / "evil.csv"
        outside.write_text("x\n1\n", encoding="utf-8")
        result = tool._run(file_path=str(outside), operation="summary")
        assert result.startswith("Error:")
        assert "outside" in result.lower()

    def test_traversal_rejected(self, tool, csv_dir):
        result = tool._run(file_path=str(csv_dir / ".." / "etc" / "passwd"), operation="summary")
        assert result.startswith("Error:")

    def test_missing_file_returns_error(self, tool, csv_dir):
        result = tool._run(file_path=str(csv_dir / "nonexistent.csv"), operation="summary")
        assert result.startswith("Error:")


class TestMalformedInput:
    def test_unknown_operation(self, tool, csv_dir):
        path = _write_csv(csv_dir, "x.csv", "a\n1\n")
        result = tool._run(file_path=path, operation="bogus")
        assert result.startswith("Error:")

    def test_aggregate_without_column(self, tool, csv_dir):
        path = _write_csv(csv_dir, "x.csv", "a\n1\n")
        result = tool._run(file_path=path, operation="aggregate")
        assert result.startswith("Error:")

    def test_filter_without_op(self, tool, csv_dir):
        path = _write_csv(csv_dir, "x.csv", "a\n1\n")
        result = tool._run(file_path=path, operation="filter", column="a")
        assert result.startswith("Error:")

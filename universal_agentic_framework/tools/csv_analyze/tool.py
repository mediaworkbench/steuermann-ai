"""CSV analysis tool — exact aggregates over workspace spreadsheet files using stdlib csv."""

import csv
import os
import re
import statistics
from pathlib import Path
from typing import Optional

import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from universal_agentic_framework.tools.vision_utils import _resolve_local_file

logger = structlog.get_logger()

_DEFAULT_WORKSPACE_ROOT = "/tmp/steuermann-ai/chat-workspaces"


def _workspace_root() -> str:
    return (
        os.environ.get("CHAT_WORKSPACE_ROOT")
        or os.environ.get("CHAT_ATTACHMENTS_ROOT")
        or _DEFAULT_WORKSPACE_ROOT
    )


def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _coerce_numeric(value: str) -> Optional[float]:
    """Tolerant numeric coercion for DE/EN formatted numbers and currency prefixes."""
    v = value.strip().lstrip("€$£¥").strip()
    # DE format: 1.234,56 → 1234.56
    if re.match(r"^-?\d{1,3}(?:\.\d{3})*,\d+$", v):
        v = v.replace(".", "").replace(",", ".")
    # EN format: 1,234.56 → 1234.56
    elif re.match(r"^-?\d{1,3}(?:,\d{3})*\.\d+$", v):
        v = v.replace(",", "")
    # Bare comma-as-decimal: 1234,56 (no thousands sep)
    elif re.match(r"^-?\d+,\d+$", v):
        v = v.replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return None


def _read_rows(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    """Return (headers, rows) where each row is a dict keyed by header."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        headers = reader.fieldnames or []
        rows = list(reader)
    return list(headers), rows


def _find_column(headers: list[str], column: str) -> Optional[str]:
    """Case-insensitive column lookup; returns canonical header name or None."""
    for h in headers:
        if h.strip().casefold() == column.strip().casefold():
            return h
    return None


def _filter_rows(
    rows: list[dict[str, str]],
    column: str,
    op: str,
    value: str,
    limit: int,
) -> tuple[list[dict[str, str]], int]:
    """Apply a simple filter; returns (matching_rows[:limit], total_matches)."""
    results: list[dict[str, str]] = []
    num_val = _coerce_numeric(value)
    for row in rows:
        cell = row.get(column, "")
        cell_num = _coerce_numeric(cell)
        match = False
        if op == "==" or op == "=":
            match = cell.strip() == value.strip() or (cell_num is not None and num_val is not None and cell_num == num_val)
        elif op == "!=":
            match = cell.strip() != value.strip()
        elif op in (">", ">=", "<", "<=") and cell_num is not None and num_val is not None:
            match = eval(f"{cell_num} {op} {num_val}")  # noqa: S307 — safe; ops are whitelisted
        elif op == "contains":
            match = value.strip().casefold() in cell.casefold()
        if match:
            results.append(row)
    total = len(results)
    return results[:limit], total


def _format_rows(headers: list[str], rows: list[dict[str, str]]) -> str:
    if not rows:
        return "(no rows)"
    col_widths = {h: max(len(h), max((len(str(r.get(h, ""))) for r in rows), default=0)) for h in headers}
    sep = " | "
    header_line = sep.join(h.ljust(col_widths[h]) for h in headers)
    divider = "-+-".join("-" * col_widths[h] for h in headers)
    data_lines = [sep.join(str(r.get(h, "")).ljust(col_widths[h]) for h in headers) for r in rows]
    return "\n".join([header_line, divider] + data_lines)


class CsvAnalyzeInput(BaseModel):
    file_path: str = Field(description="Absolute path to the CSV workspace document.")
    operation: str = Field(
        description=(
            "Operation to perform: summary | aggregate | filter | head | tail | unique | value_counts. "
            "Use 'summary' for an overview; 'aggregate' for sum/mean/min/max/count; "
            "'filter' to find rows matching a condition; 'head'/'tail' for first/last rows; "
            "'unique'/'value_counts' for distinct values in a column."
        )
    )
    column: Optional[str] = Field(default=None, description="Column name (required for aggregate, filter, unique, value_counts).")
    aggregation: Optional[str] = Field(default=None, description="Aggregation function: sum | mean | min | max | count (used with 'aggregate' operation).")
    group_by: Optional[str] = Field(default=None, description="Optional column to group by when using 'aggregate'.")
    filter_op: Optional[str] = Field(default=None, description="Filter operator: == != > >= < <= contains (used with 'filter' operation).")
    filter_value: Optional[str] = Field(default=None, description="Value to compare against in 'filter' operation.")
    limit: int = Field(default=20, description="Maximum number of rows to return for filter/head/tail/value_counts.")


class CsvAnalyzeTool(BaseTool):
    """Analyze a CSV workspace document — compute exact aggregates over all rows using stdlib csv."""

    name: str = "csv_analyze_tool"
    description: str = (
        "Analyze a CSV spreadsheet file from the workspace. "
        "Supported operations: summary (row count, columns, types), "
        "aggregate (sum / total / average / mean / min / max / count) over a column, "
        "with optional group by; filter rows by column value; "
        "head / tail (first / last rows); unique values; value_counts. "
        "Use this when the user asks to sum, total, average, count rows, "
        "how many rows, group by, filter a spreadsheet or CSV column. "
        "Trigger phrases: sum column, total amount, average price, count rows, "
        "how many entries, group by category, filter where, show first rows, "
        "Summe, Durchschnitt, Mittelwert, wie viele Zeilen, Spalte auswerten, "
        "CSV auswerten, Tabelle auswerten, gruppieren, filtern, nach Spalte."
    )
    args_schema: type[BaseModel] = CsvAnalyzeInput

    workspace_base_dir: str = Field(default_factory=_workspace_root)

    def _run(
        self,
        file_path: str,
        operation: str,
        column: Optional[str] = None,
        aggregation: Optional[str] = None,
        group_by: Optional[str] = None,
        filter_op: Optional[str] = None,
        filter_value: Optional[str] = None,
        limit: int = 20,
        **kwargs,
    ) -> str:
        try:
            path = _resolve_local_file(file_path, self.workspace_base_dir)
        except (ValueError, FileNotFoundError) as exc:
            return f"Error: {exc}"

        try:
            sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
        except OSError as exc:
            return f"Error reading file: {exc}"

        delimiter = _sniff_delimiter(sample)

        try:
            headers, rows = _read_rows(path, delimiter)
        except Exception as exc:
            return f"Error parsing CSV: {exc}"

        if not headers:
            return "Error: CSV has no headers."

        op = operation.strip().lower()

        if op == "summary":
            return self._op_summary(headers, rows)
        if op == "aggregate":
            return self._op_aggregate(headers, rows, column, aggregation, group_by)
        if op == "filter":
            return self._op_filter(headers, rows, column, filter_op, filter_value, limit)
        if op == "head":
            return _format_rows(headers, rows[:limit])
        if op == "tail":
            return _format_rows(headers, rows[-limit:])
        if op == "unique":
            return self._op_unique(headers, rows, column)
        if op == "value_counts":
            return self._op_value_counts(headers, rows, column, limit)
        return f"Error: Unknown operation '{operation}'. Valid: summary, aggregate, filter, head, tail, unique, value_counts."

    async def _arun(
        self,
        file_path: str,
        operation: str,
        column: Optional[str] = None,
        aggregation: Optional[str] = None,
        group_by: Optional[str] = None,
        filter_op: Optional[str] = None,
        filter_value: Optional[str] = None,
        limit: int = 20,
        **kwargs,
    ) -> str:
        return self._run(
            file_path=file_path,
            operation=operation,
            column=column,
            aggregation=aggregation,
            group_by=group_by,
            filter_op=filter_op,
            filter_value=filter_value,
            limit=limit,
        )

    # ── operations ──────────────────────────────────────────────────────────

    def _op_summary(self, headers: list[str], rows: list[dict]) -> str:
        lines = [
            f"Rows: {len(rows)}",
            f"Columns ({len(headers)}): {', '.join(headers)}",
            "",
            "Column types (inferred):",
        ]
        for h in headers:
            values = [row.get(h, "") for row in rows if row.get(h, "").strip()]
            numeric_count = sum(1 for v in values if _coerce_numeric(v) is not None)
            pct = (numeric_count / len(values) * 100) if values else 0
            col_type = "numeric" if pct >= 80 else "text"
            lines.append(f"  {h}: {col_type} ({numeric_count}/{len(values)} numeric values)")
        return "\n".join(lines)

    def _op_aggregate(
        self,
        headers: list[str],
        rows: list[dict],
        column: Optional[str],
        aggregation: Optional[str],
        group_by: Optional[str],
    ) -> str:
        if not column:
            return "Error: 'column' is required for aggregate operation."
        agg = (aggregation or "sum").strip().lower()
        canonical = _find_column(headers, column)
        if not canonical:
            return f"Error: Column '{column}' not found. Available: {', '.join(headers)}"

        if group_by:
            gb_canonical = _find_column(headers, group_by)
            if not gb_canonical:
                return f"Error: group_by column '{group_by}' not found. Available: {', '.join(headers)}"
            # count tracks non-empty strings per group; numeric_groups tracks parseable floats.
            group_counts: dict[str, int] = {}
            numeric_groups: dict[str, list[float]] = {}
            for row in rows:
                key = row.get(gb_canonical, "").strip()
                cell = row.get(canonical, "")
                if cell.strip():
                    group_counts[key] = group_counts.get(key, 0) + 1
                num = _coerce_numeric(cell)
                if num is not None:
                    numeric_groups.setdefault(key, []).append(num)
            if agg == "count":
                if not group_counts:
                    return f"No non-empty values found in column '{canonical}'."
                result_lines = [f"count({canonical}) group by {gb_canonical}"]
                for key in sorted(group_counts):
                    result_lines.append(f"  {key}: {group_counts[key]}")
                return "\n".join(result_lines)
            if not numeric_groups:
                return f"No numeric values found in column '{canonical}'."
            result_lines = [f"Aggregation: {agg}({canonical}) group by {gb_canonical}"]
            for key in sorted(numeric_groups):
                vals = numeric_groups[key]
                result_lines.append(f"  {key}: {self._apply_agg(agg, vals)}")
            return "\n".join(result_lines)

        # No group_by
        if agg == "count":
            non_empty = sum(1 for row in rows if row.get(canonical, "").strip())
            return f"count({canonical}): {non_empty}"
        vals = [_coerce_numeric(row.get(canonical, "")) for row in rows]
        nums = [v for v in vals if v is not None]
        if not nums:
            return f"No numeric values found in column '{canonical}'."
        return f"{agg}({canonical}): {self._apply_agg(agg, nums)}"

    @staticmethod
    def _fmt(value: float) -> str:
        """Format a float, stripping insignificant trailing zeros after rounding."""
        return str(round(value, 6)).rstrip("0").rstrip(".")

    @staticmethod
    def _apply_agg(agg: str, nums: list[float]) -> str:
        if agg == "sum":
            return CsvAnalyzeTool._fmt(sum(nums))
        if agg in ("mean", "average", "avg"):
            return CsvAnalyzeTool._fmt(statistics.mean(nums))
        if agg == "min":
            return CsvAnalyzeTool._fmt(min(nums))
        if agg == "max":
            return CsvAnalyzeTool._fmt(max(nums))
        if agg == "count":
            return str(len(nums))
        return f"Unknown aggregation '{agg}'"

    def _op_filter(
        self,
        headers: list[str],
        rows: list[dict],
        column: Optional[str],
        filter_op: Optional[str],
        filter_value: Optional[str],
        limit: int,
    ) -> str:
        if not column:
            return "Error: 'column' is required for filter operation."
        if not filter_op:
            return "Error: 'filter_op' is required for filter operation (== != > >= < <= contains)."
        if filter_value is None:
            return "Error: 'filter_value' is required for filter operation."
        canonical = _find_column(headers, column)
        if not canonical:
            return f"Error: Column '{column}' not found. Available: {', '.join(headers)}"
        matched, total = _filter_rows(rows, canonical, filter_op, filter_value, limit)
        header_line = f"Filter: {canonical} {filter_op} '{filter_value}' → {total} matching rows (showing up to {limit})\n"
        return header_line + _format_rows(headers, matched)

    def _op_unique(self, headers: list[str], rows: list[dict], column: Optional[str]) -> str:
        if not column:
            return "Error: 'column' is required for unique operation."
        canonical = _find_column(headers, column)
        if not canonical:
            return f"Error: Column '{column}' not found. Available: {', '.join(headers)}"
        seen: dict[str, int] = {}
        for row in rows:
            v = row.get(canonical, "").strip()
            seen[v] = seen.get(v, 0) + 1
        values = sorted(seen)
        return f"Unique values in '{canonical}' ({len(values)} distinct):\n" + "\n".join(values)

    def _op_value_counts(
        self, headers: list[str], rows: list[dict], column: Optional[str], limit: int
    ) -> str:
        if not column:
            return "Error: 'column' is required for value_counts operation."
        canonical = _find_column(headers, column)
        if not canonical:
            return f"Error: Column '{column}' not found. Available: {', '.join(headers)}"
        counts: dict[str, int] = {}
        for row in rows:
            v = row.get(canonical, "").strip()
            counts[v] = counts.get(v, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:limit]
        lines = [f"Value counts for '{canonical}' (top {limit}):"]
        for val, cnt in sorted_counts:
            lines.append(f"  {val!r}: {cnt}")
        return "\n".join(lines)

"""Smoke tests for the steuermann CLI surface."""

import json
from pathlib import Path

import pytest

import universal_agentic_framework.cli.steuermann as steuermann


def test_parser_has_expected_top_level_commands() -> None:
    parser = steuermann.create_parser()
    args = parser.parse_args(["profile", "active"])
    assert args.command == "profile"
    assert args.profile_command == "active"


def test_ingest_group_has_reindex_subcommand() -> None:
    parser = steuermann.create_parser()
    args = parser.parse_args(
        [
            "ingest",
            "reindex",
            "--source",
            "data/rag-data",
            "--yes",
        ]
    )
    assert args.command == "ingest"
    assert args.ingest_command == "reindex"


def test_docs_check_command_runs_non_strict() -> None:
    code = steuermann.main(["docs", "check", "--format", "json"])
    assert code == 0


def test_docs_check_strict_fails_on_missing_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    code = steuermann.main(["docs", "check", "--strict", "--format", "json"])
    assert code == 1


def test_setup_doctor_fails_when_postgres_password_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    code = steuermann.main(["setup", "doctor", "--format", "json"])
    assert code == 1


def test_config_explain_emits_source_chain(capsys: pytest.CaptureFixture[str]) -> None:
    code = steuermann.main(
        [
            "config",
            "explain",
            "--key",
            "core.rag.collection_name",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["key"] == "core.rag.collection_name"
    assert isinstance(payload["source_chain"], list)
    assert payload["source_chain"]

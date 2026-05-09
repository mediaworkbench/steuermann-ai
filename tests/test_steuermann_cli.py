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


def test_docs_check_reports_contract_parity_details(capsys: pytest.CaptureFixture[str]) -> None:
    code = steuermann.main(["docs", "check", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    details = [check["details"] for check in payload["checks"]]
    assert any("Contract covers expected CLI/runtime config sections" in detail for detail in details)
    assert any("Precedence matches base -> profile -> environment" in detail for detail in details)


def test_config_contract_check_runs(capsys: pytest.CaptureFixture[str]) -> None:
    code = steuermann.main(["config", "contract-check", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"


def test_config_validate_strict_fails_on_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        steuermann,
        "_validate_one_profile",
        lambda profile_id: {"profile": profile_id, "status": "ok", "errors": [], "warnings": ["warning"]},
    )
    monkeypatch.setattr(steuermann, "_iter_profiles", lambda: [])
    code = steuermann.main(["config", "validate", "--strict", "--format", "json"])
    assert code == 1


def test_docs_check_strict_fails_on_contract_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "config" / "contracts").mkdir(parents=True)
    (tmp_path / "README.md").write_text("steuermann\n", encoding="utf-8")
    (tmp_path / "docs" / "index.md").write_text("steuermann\n", encoding="utf-8")
    (tmp_path / "docs" / "configuration.md").write_text(
        "repository defaults\nprofile overlay\nenvironment variables\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "profile_creation.md").write_text("profile\n", encoding="utf-8")
    (tmp_path / "config" / "contracts" / "cli_contract.yaml").write_text(
        "schema_version: 1\nprecedence: [base, environment]\nsections: {}\npolicies:\n  docs_mutation: disabled\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = steuermann.main(["docs", "check", "--strict", "--format", "json"])
    assert code == 1

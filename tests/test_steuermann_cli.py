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


def test_setup_doctor_fails_when_postgres_password_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    code = steuermann.main(["setup", "doctor", "--format", "json"])
    assert code == 1


def test_setup_doctor_reports_env_presence_and_advisories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / ".env").write_text(
        "POSTGRES_PASSWORD=secret\nLLM_ENDPOINT=http://localhost:1234/v1\nEMBEDDING_SERVER=http://localhost:1234/v1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    class DummyRag:
        collection_name = "framework"

    class DummyCore:
        rag = DummyRag()

    monkeypatch.setattr(steuermann, "load_core_config", lambda env=None: DummyCore())

    code = steuermann.main(["setup", "doctor", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    names = [check["name"] for check in payload["checks"]]
    assert ".env presence" in names
    assert "LLM_ENDPOINT" in names
    assert "EMBEDDING_SERVER" in names


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
    details = [check["details"] for check in payload["checks"]]
    assert any("core.profile_overlay_file matches" in detail for detail in details)
    assert any("severity_policy.blocking is error" in detail for detail in details)
    assert any("allowed_core_prefixes matches loader constants" in detail for detail in details)
    assert any("disallowed_feature_flags matches loader constants" in detail for detail in details)


def test_config_contract_check_detects_prefix_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "config" / "contracts").mkdir(parents=True)
    bad_contract = {
        "schema_version": 1,
        "contract_name": "test",
        "precedence": ["base", "profile", "environment"],
        "sections": {s: {"source_file": f"config/{f}", "profile_overlay_file": f"config/profiles/<profile_id>/{f}", "profile_mutability": "partial"} for s, f in {"core": "core.yaml", "features": "features.yaml", "tools": "tools.yaml", "agents": "agents.yaml"}.items()},
        "policies": {"docs_mutation": "disabled", "manual_config_editing": "supported", "ingest_interface": "steuermann_ingest_only", "json_output_stability": "required"},
        "severity_policy": {"blocking": "error", "advisory": "warning"},
        "profile_safety": {
            "allowed_core_prefixes": ["llm", "prompts"],  # incomplete
            "disallowed_feature_flags": ["authentication", "ingestion_service", "monitoring"],
        },
    }
    import yaml as _yaml
    (tmp_path / "config" / "contracts" / "cli_contract.yaml").write_text(
        _yaml.safe_dump(bad_contract), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    code = steuermann.main(["config", "contract-check", "--format", "json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    details = [check["details"] for check in payload["checks"]]
    assert any("allowed_core_prefixes drift" in d for d in details)


def test_config_contract_check_detects_flag_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "config" / "contracts").mkdir(parents=True)
    bad_contract = {
        "schema_version": 1,
        "contract_name": "test",
        "precedence": ["base", "profile", "environment"],
        "sections": {s: {"source_file": f"config/{f}", "profile_overlay_file": f"config/profiles/<profile_id>/{f}", "profile_mutability": "partial"} for s, f in {"core": "core.yaml", "features": "features.yaml", "tools": "tools.yaml", "agents": "agents.yaml"}.items()},
        "policies": {"docs_mutation": "disabled", "manual_config_editing": "supported", "ingest_interface": "steuermann_ingest_only", "json_output_stability": "required"},
        "severity_policy": {"blocking": "error", "advisory": "warning"},
        "profile_safety": {
            "allowed_core_prefixes": ["fork.language", "fork.locale", "fork.timezone", "fork.supported_languages", "llm", "prompts", "tool_routing", "rag", "tokens", "memory.retention"],
            "disallowed_feature_flags": ["authentication"],  # missing ingestion_service, monitoring
        },
    }
    import yaml as _yaml
    (tmp_path / "config" / "contracts" / "cli_contract.yaml").write_text(
        _yaml.safe_dump(bad_contract), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    code = steuermann.main(["config", "contract-check", "--format", "json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    details = [check["details"] for check in payload["checks"]]
    assert any("disallowed_feature_flags drift" in d for d in details)


def test_config_validate_strict_fails_on_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        steuermann,
        "_validate_one_profile",
        lambda profile_id: {"profile": profile_id, "status": "ok", "errors": [], "warnings": ["warning"]},
    )
    monkeypatch.setattr(steuermann, "_iter_profiles", lambda: [])
    code = steuermann.main(["config", "validate", "--strict", "--format", "json"])
    assert code == 1


def test_load_env_file_reads_dotenv_without_overriding_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text("FOO=from-file\nBAR=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BAR", "from-env")
    env = steuermann._load_env_file()
    assert env["FOO"] == "from-file"
    assert env["BAR"] == "from-env"


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

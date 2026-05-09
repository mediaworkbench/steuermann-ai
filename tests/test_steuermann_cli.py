"""Smoke tests for the steuermann CLI surface."""

import json
from pathlib import Path
import sys
import tarfile

import pytest
import yaml

import universal_agentic_framework.cli.steuermann as steuermann


def _create_profile_dir(root: Path, name: str = "starter") -> Path:
    profile_dir = root / "config" / "profiles" / name
    profile_dir.mkdir(parents=True)
    (profile_dir / "prompts").mkdir()
    for filename in ["core.yaml", "features.yaml", "agents.yaml", "tools.yaml", "ui.yaml"]:
        (profile_dir / filename).write_text("{}\n", encoding="utf-8")
    (profile_dir / "profile.yaml").write_text(
        "profile_id: starter\n"
        "display_name: Starter\n"
        "description: Starter profile\n",
        encoding="utf-8",
    )
    return profile_dir


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
    monkeypatch.delenv("PROFILE_ID", raising=False)
    monkeypatch.delenv("ACTIVE_PROFILE_ID", raising=False)

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


def test_config_set_dry_run_does_not_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    core_path = profile_dir / "core.yaml"
    core_path.write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    before = core_path.read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = steuermann.main(
        [
            "config",
            "set",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--value",
            "0.4",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["mode"] == "dry-run"
    assert payload["new_value"] == 0.4
    assert core_path.read_text(encoding="utf-8") == before


def test_config_set_apply_writes_and_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    core_path = profile_dir / "core.yaml"
    core_path.write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        steuermann,
        "_validate_one_profile",
        lambda profile_id: {"profile": profile_id, "status": "ok", "errors": [], "warnings": []},
    )

    code = steuermann.main(
        [
            "config",
            "set",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--value",
            "0.5",
            "--apply",
            "--confirm",
            "APPLY",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["mode"] == "apply"
    assert payload["reverted"] is False
    data = yaml.safe_load(core_path.read_text(encoding="utf-8"))
    assert data["llm"]["temperature"] == 0.5


def test_config_set_rejects_non_profile_safe_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _create_profile_dir(tmp_path)
    monkeypatch.chdir(tmp_path)

    code = steuermann.main(
        [
            "config",
            "set",
            "--profile",
            "starter",
            "--key",
            "core.memory.vector_store.host",
            "--value",
            "qdrant",
            "--format",
            "json",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "profile-safe" in payload["error"]


def test_config_set_apply_reverts_on_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    core_path = profile_dir / "core.yaml"
    core_path.write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    before = core_path.read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        steuermann,
        "_validate_one_profile",
        lambda profile_id: {"profile": profile_id, "status": "error", "errors": ["bad"], "warnings": []},
    )

    code = steuermann.main(
        [
            "config",
            "set",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--value",
            "0.1",
            "--apply",
            "--confirm",
            "APPLY",
            "--format",
            "json",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["reverted"] is True
    assert core_path.read_text(encoding="utf-8") == before


def test_config_unset_dry_run_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    core_path = profile_dir / "core.yaml"
    core_path.write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    before = core_path.read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = steuermann.main(
        [
            "config",
            "unset",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["mode"] == "dry-run"
    assert payload["would_change"] is True
    assert core_path.read_text(encoding="utf-8") == before


def test_config_unset_apply_removes_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    core_path = profile_dir / "core.yaml"
    core_path.write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        steuermann,
        "_validate_one_profile",
        lambda profile_id: {"profile": profile_id, "status": "ok", "errors": [], "warnings": []},
    )

    code = steuermann.main(
        [
            "config",
            "unset",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--apply",
            "--confirm",
            "APPLY",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["mode"] == "apply"
    assert payload["reverted"] is False
    data = yaml.safe_load(core_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "llm" not in data


def test_config_unset_apply_reverts_on_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    core_path = profile_dir / "core.yaml"
    core_path.write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    before = core_path.read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        steuermann,
        "_validate_one_profile",
        lambda profile_id: {"profile": profile_id, "status": "error", "errors": ["bad"], "warnings": []},
    )

    code = steuermann.main(
        [
            "config",
            "unset",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--apply",
            "--confirm",
            "APPLY",
            "--format",
            "json",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["reverted"] is True
    assert core_path.read_text(encoding="utf-8") == before


def test_docs_check_reports_contract_parity_details(capsys: pytest.CaptureFixture[str]) -> None:
    code = steuermann.main(["docs", "check", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    details = [check["details"] for check in payload["checks"]]
    assert any("Contract covers expected CLI/runtime config sections" in detail for detail in details)
    assert any("Precedence matches base -> profile -> environment" in detail for detail in details)
    assert "drift_report" in payload
    assert "total" in payload["drift_report"]
    assert "items" in payload["drift_report"]
    assert "by_domain" in payload["drift_report"]


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
    assert any("framework_version_range_default matches CLI default" in detail for detail in details)
    assert any("minimum_required_keys_default matches CLI defaults" in detail for detail in details)
    assert any("mutator_surface.config_set.profile_scope is named_profile_only" in detail for detail in details)
    assert any("mutator_surface.config_set.key_scope is profile_safe_core_only" in detail for detail in details)
    assert any("mutator_surface.config_set.requires_confirm_token is True" in detail for detail in details)
    assert any("mutator_surface.config_set.confirm_token is APPLY" in detail for detail in details)
    assert any("mutator_surface.config_set.interactive_tty_prompt_fallback is True" in detail for detail in details)
    assert any("mutator_surface.config_set.non_interactive_requires_confirm_flag is True" in detail for detail in details)
    assert any("mutator_surface.config_unset.profile_scope is named_profile_only" in detail for detail in details)
    assert any("mutator_surface.config_unset.key_scope is profile_safe_core_only" in detail for detail in details)
    assert any("mutator_surface.config_unset.requires_confirm_token is True" in detail for detail in details)
    assert any("mutator_surface.config_unset.confirm_token is APPLY" in detail for detail in details)
    assert any("mutator_surface.config_unset.interactive_tty_prompt_fallback is True" in detail for detail in details)
    assert any("mutator_surface.config_unset.non_interactive_requires_confirm_flag is True" in detail for detail in details)


def test_config_set_apply_requires_confirm_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    (profile_dir / "core.yaml").write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    code = steuermann.main(
        [
            "config",
            "set",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--value",
            "0.5",
            "--apply",
            "--format",
            "json",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "--confirm APPLY" in payload["error"]


def test_config_unset_apply_requires_confirm_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    (profile_dir / "core.yaml").write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    code = steuermann.main(
        [
            "config",
            "unset",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--apply",
            "--format",
            "json",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "--confirm APPLY" in payload["error"]


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


def test_config_contract_check_detects_bundle_compatibility_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "config" / "contracts").mkdir(parents=True)
    bad_contract = {
        "schema_version": 1,
        "contract_name": "test",
        "precedence": ["base", "profile", "environment"],
        "sections": {
            s: {
                "source_file": f"config/{f}",
                "profile_overlay_file": f"config/profiles/<profile_id>/{f}",
                "profile_mutability": "partial",
            }
            for s, f in {"core": "core.yaml", "features": "features.yaml", "tools": "tools.yaml", "agents": "agents.yaml"}.items()
        },
        "policies": {
            "docs_mutation": "disabled",
            "manual_config_editing": "supported",
            "ingest_interface": "steuermann_ingest_only",
            "json_output_stability": "required",
        },
        "severity_policy": {"blocking": "error", "advisory": "warning"},
        "profile_safety": {
            "allowed_core_prefixes": [
                "fork.language",
                "fork.locale",
                "fork.timezone",
                "fork.supported_languages",
                "llm",
                "prompts",
                "tool_routing",
                "rag",
                "tokens",
                "memory.retention",
            ],
            "disallowed_feature_flags": ["authentication", "ingestion_service", "monitoring"],
        },
        "profile_bundle_compatibility": {
            "framework_version_range_default": ">=9.0,<10.0",
            "minimum_required_keys_default": ["profile_id"],
        },
    }
    (tmp_path / "config" / "contracts" / "cli_contract.yaml").write_text(
        yaml.safe_dump(bad_contract), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    code = steuermann.main(["config", "contract-check", "--format", "json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    details = [check["details"] for check in payload["checks"]]
    assert any("framework_version_range_default mismatch" in d for d in details)
    assert any("minimum_required_keys_default drift" in d for d in details)


def test_config_contract_check_detects_mutator_surface_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "config" / "contracts").mkdir(parents=True)
    bad_contract = {
        "schema_version": 1,
        "contract_name": "test",
        "precedence": ["base", "profile", "environment"],
        "sections": {
            s: {
                "source_file": f"config/{f}",
                "profile_overlay_file": f"config/profiles/<profile_id>/{f}",
                "profile_mutability": "partial",
            }
            for s, f in {"core": "core.yaml", "features": "features.yaml", "tools": "tools.yaml", "agents": "agents.yaml"}.items()
        },
        "policies": {
            "docs_mutation": "disabled",
            "manual_config_editing": "supported",
            "ingest_interface": "steuermann_ingest_only",
            "json_output_stability": "required",
        },
        "severity_policy": {"blocking": "error", "advisory": "warning"},
        "profile_safety": {
            "allowed_core_prefixes": [
                "fork.language",
                "fork.locale",
                "fork.timezone",
                "fork.supported_languages",
                "llm",
                "prompts",
                "tool_routing",
                "rag",
                "tokens",
                "memory.retention",
            ],
            "disallowed_feature_flags": ["authentication", "ingestion_service", "monitoring"],
        },
        "profile_bundle_compatibility": {
            "framework_version_range_default": steuermann.DEFAULT_BUNDLE_FRAMEWORK_VERSION_RANGE,
            "minimum_required_keys_default": steuermann.DEFAULT_BUNDLE_REQUIRED_KEYS,
        },
        "mutator_surface": {
            "config_set": {
                "profile_scope": "base_allowed",
                "target_file": "features.yaml",
                "key_scope": "any",
                "dry_run_default": False,
                "requires_apply_flag": False,
                "requires_confirm_token": False,
                "confirm_token": "NOPE",
                "interactive_tty_prompt_fallback": False,
                "non_interactive_requires_confirm_flag": False,
                "rollback_on_validation_error": False,
            },
            "config_unset": {
                "profile_scope": "base_allowed",
                "target_file": "features.yaml",
                "key_scope": "any",
                "dry_run_default": False,
                "requires_apply_flag": False,
                "requires_confirm_token": False,
                "confirm_token": "NOPE",
                "interactive_tty_prompt_fallback": False,
                "non_interactive_requires_confirm_flag": False,
                "rollback_on_validation_error": False,
            },
        },
    }
    (tmp_path / "config" / "contracts" / "cli_contract.yaml").write_text(
        yaml.safe_dump(bad_contract), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    code = steuermann.main(["config", "contract-check", "--format", "json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    details = [check["details"] for check in payload["checks"]]
    assert any("mutator_surface.config_set.profile_scope mismatch" in d for d in details)
    assert any("mutator_surface.config_set.key_scope mismatch" in d for d in details)
    assert any("mutator_surface.config_set.requires_confirm_token mismatch" in d for d in details)
    assert any("mutator_surface.config_set.confirm_token mismatch" in d for d in details)
    assert any("mutator_surface.config_set.interactive_tty_prompt_fallback mismatch" in d for d in details)
    assert any("mutator_surface.config_set.non_interactive_requires_confirm_flag mismatch" in d for d in details)
    assert any("mutator_surface.config_unset.profile_scope mismatch" in d for d in details)
    assert any("mutator_surface.config_unset.key_scope mismatch" in d for d in details)
    assert any("mutator_surface.config_unset.requires_confirm_token mismatch" in d for d in details)
    assert any("mutator_surface.config_unset.confirm_token mismatch" in d for d in details)
    assert any("mutator_surface.config_unset.interactive_tty_prompt_fallback mismatch" in d for d in details)
    assert any("mutator_surface.config_unset.non_interactive_requires_confirm_flag mismatch" in d for d in details)


def test_config_set_apply_accepts_interactive_confirm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    core_path = profile_dir / "core.yaml"
    core_path.write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "APPLY")
    monkeypatch.setattr(
        steuermann,
        "_validate_one_profile",
        lambda profile_id: {"profile": profile_id, "status": "ok", "errors": [], "warnings": []},
    )

    code = steuermann.main(
        [
            "config",
            "set",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--value",
            "0.2",
            "--apply",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["confirmation_source"] == "interactive"
    data = yaml.safe_load(core_path.read_text(encoding="utf-8"))
    assert data["llm"]["temperature"] == 0.2


def test_config_unset_apply_accepts_interactive_confirm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    core_path = profile_dir / "core.yaml"
    core_path.write_text("llm:\n  temperature: 0.7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "APPLY")
    monkeypatch.setattr(
        steuermann,
        "_validate_one_profile",
        lambda profile_id: {"profile": profile_id, "status": "ok", "errors": [], "warnings": []},
    )

    code = steuermann.main(
        [
            "config",
            "unset",
            "--profile",
            "starter",
            "--key",
            "core.llm.temperature",
            "--apply",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["confirmation_source"] == "interactive"
    data = yaml.safe_load(core_path.read_text(encoding="utf-8"))
    assert "llm" not in data


def test_profile_bundle_import_does_not_modify_existing_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starter_dir = _create_profile_dir(tmp_path, name="starter")
    (starter_dir / "profile.yaml").write_text(
        "profile_id: starter\n"
        "display_name: Starter Original\n"
        "description: Original starter profile\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(steuermann, "_framework_version", lambda: "0.2")

    bundle_path = tmp_path / "starter.tar.gz"
    export_code = steuermann.main(
        [
            "profile",
            "bundle",
            "export",
            "--profile",
            "starter",
            "--out",
            str(bundle_path),
            "--format",
            "json",
        ]
    )
    assert export_code == 0

    import_code = steuermann.main(
        [
            "profile",
            "bundle",
            "import",
            "--bundle",
            str(bundle_path),
            "--profile",
            "imported-starter",
            "--format",
            "json",
        ]
    )
    assert import_code == 0

    assert (tmp_path / "config" / "profiles" / "starter").exists()
    assert (tmp_path / "config" / "profiles" / "imported-starter").exists()

    starter_profile = yaml.safe_load(
        (tmp_path / "config" / "profiles" / "starter" / "profile.yaml").read_text(encoding="utf-8")
    )
    assert starter_profile["display_name"] == "Starter Original"

    imported_profile = yaml.safe_load(
        (tmp_path / "config" / "profiles" / "imported-starter" / "profile.yaml").read_text(encoding="utf-8")
    )
    assert imported_profile["profile_id"] == "imported-starter"
    assert steuermann._validate_profile_files(tmp_path / "config" / "profiles" / "imported-starter") == []


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


def test_docs_check_reports_drift_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "config" / "contracts").mkdir(parents=True)
    (tmp_path / "README.md").write_text("missing steuermann references\n", encoding="utf-8")
    (tmp_path / "docs" / "index.md").write_text("missing command index\n", encoding="utf-8")
    (tmp_path / "docs" / "configuration.md").write_text("missing precedence\n", encoding="utf-8")
    (tmp_path / "docs" / "profile_creation.md").write_text("profile\n", encoding="utf-8")
    (tmp_path / "config" / "contracts" / "cli_contract.yaml").write_text(
        "schema_version: 1\nprecedence: [base, profile, environment]\nsections: {}\npolicies:\n  docs_mutation: disabled\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    code = steuermann.main(["docs", "check", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["drift_report"]["total"] > 0
    assert payload["drift_report"]["items"]
    assert any(item["status"] == "fail" for item in payload["drift_report"]["items"])
    assert all("domain" in item for item in payload["drift_report"]["items"])
    assert "contract" in payload["drift_report"]["by_domain"]


def test_docs_check_classifies_bundle_compat_drift_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
        yaml.safe_dump(
            {
                "schema_version": 1,
                "precedence": ["base", "profile", "environment"],
                "sections": {
                    "core": {
                        "source_file": "config/core.yaml",
                        "profile_overlay_file": "config/profiles/<profile_id>/core.yaml",
                        "profile_mutability": "partial",
                    },
                    "features": {
                        "source_file": "config/features.yaml",
                        "profile_overlay_file": "config/profiles/<profile_id>/features.yaml",
                        "profile_mutability": "partial",
                    },
                    "tools": {
                        "source_file": "config/tools.yaml",
                        "profile_overlay_file": "config/profiles/<profile_id>/tools.yaml",
                        "profile_mutability": "partial",
                    },
                    "agents": {
                        "source_file": "config/agents.yaml",
                        "profile_overlay_file": "config/profiles/<profile_id>/agents.yaml",
                        "profile_mutability": "partial",
                    },
                },
                "policies": {
                    "docs_mutation": "disabled",
                    "manual_config_editing": "supported",
                    "ingest_interface": "steuermann_ingest_only",
                    "json_output_stability": "required",
                },
                "severity_policy": {"blocking": "error", "advisory": "warning"},
                "profile_safety": {
                    "allowed_core_prefixes": [
                        "fork.language",
                        "fork.locale",
                        "fork.timezone",
                        "fork.supported_languages",
                        "llm",
                        "prompts",
                        "tool_routing",
                        "rag",
                        "tokens",
                        "memory.retention",
                    ],
                    "disallowed_feature_flags": ["authentication", "ingestion_service", "monitoring"],
                },
                "profile_bundle_compatibility": {
                    "framework_version_range_default": ">=9.0,<10.0",
                    "minimum_required_keys_default": steuermann.DEFAULT_BUNDLE_REQUIRED_KEYS,
                },
            },
            sort_keys=True,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = steuermann.main(["docs", "check", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["drift_report"]["by_domain"]["bundle-compat"] >= 1
    assert any(item["domain"] == "bundle-compat" for item in payload["drift_report"]["items"])


def test_profile_scaffold_writes_compatibility_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _create_profile_dir(tmp_path)
    monkeypatch.chdir(tmp_path)

    target = tmp_path / "config" / "profiles" / "external-profile"
    code = steuermann.main(
        [
            "profile",
            "scaffold",
            "--from",
            "starter",
            "--profile",
            "external-profile",
            "--format",
            "json",
        ]
    )
    assert code == 0

    # Compatibility metadata should NOT be in profile.yaml
    profile_data = yaml.safe_load((target / "profile.yaml").read_text(encoding="utf-8"))
    assert "compatibility" not in profile_data, "profile.yaml should not contain compatibility metadata"
    assert profile_data.get("profile_id") == "external-profile"
    assert profile_data.get("display_name") is not None

    # Compatibility metadata should be in bundle_manifest.yaml
    manifest = yaml.safe_load((target / "bundle_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest.get("compatibility", {}).get("framework_version_range") == steuermann.DEFAULT_BUNDLE_FRAMEWORK_VERSION_RANGE
    assert manifest.get("compatibility", {}).get("minimum_required_keys") == steuermann.DEFAULT_BUNDLE_REQUIRED_KEYS


def test_profile_bundle_import_fails_on_incompatible_major_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_profile_dir(tmp_path)
    monkeypatch.chdir(tmp_path)

    bundle_path = tmp_path / "starter.tar.gz"
    export_code = steuermann.main(
        [
            "profile",
            "bundle",
            "export",
            "--profile",
            "starter",
            "--out",
            str(bundle_path),
            "--format",
            "json",
        ]
    )
    assert export_code == 0

    # Rewrite bundle manifest to require incompatible major framework version.
    with tarfile.open(bundle_path, "r:gz") as src:
        members = src.getmembers()
        files: dict[str, bytes] = {}
        for member in members:
            extracted = src.extractfile(member)
            if extracted is not None:
                files[member.name] = extracted.read()

    manifest = yaml.safe_load(files["bundle_manifest.yaml"].decode("utf-8"))
    manifest["compatibility"]["framework_version_range"] = ">=1.0,<2.0"
    files["bundle_manifest.yaml"] = yaml.safe_dump(manifest, sort_keys=True, allow_unicode=False).encode("utf-8")

    with tarfile.open(bundle_path, "w:gz") as dst:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            dst.addfile(info, steuermann._BytesReader(content))

    target = tmp_path / "imported-profile"
    code = steuermann.main(
        [
            "profile",
            "bundle",
            "import",
            "--bundle",
            str(bundle_path),
            "--profile",
            "imported-profile",
            "--format",
            "json",
        ]
    )
    assert code == 1
    assert not target.exists()


def test_profile_bundle_import_fails_when_required_profile_keys_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_profile_dir(tmp_path)
    monkeypatch.chdir(tmp_path)

    bundle_path = tmp_path / "starter.tar.gz"
    export_code = steuermann.main(
        [
            "profile",
            "bundle",
            "export",
            "--profile",
            "starter",
            "--out",
            str(bundle_path),
            "--format",
            "json",
        ]
    )
    assert export_code == 0

    with tarfile.open(bundle_path, "r:gz") as src:
        members = src.getmembers()
        files: dict[str, bytes] = {}
        for member in members:
            extracted = src.extractfile(member)
            if extracted is not None:
                files[member.name] = extracted.read()

    # Remove required field from profile and make manifest require it.
    profile_data = yaml.safe_load(files["starter/profile.yaml"].decode("utf-8"))
    profile_data.pop("description", None)
    files["starter/profile.yaml"] = yaml.safe_dump(profile_data, sort_keys=True, allow_unicode=False).encode("utf-8")

    manifest = yaml.safe_load(files["bundle_manifest.yaml"].decode("utf-8"))
    manifest["compatibility"]["minimum_required_keys"] = ["profile_id", "display_name", "description"]
    files["bundle_manifest.yaml"] = yaml.safe_dump(manifest, sort_keys=True, allow_unicode=False).encode("utf-8")

    with tarfile.open(bundle_path, "w:gz") as dst:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            dst.addfile(info, steuermann._BytesReader(content))

    target = tmp_path / "imported-profile"
    code = steuermann.main(
        [
            "profile",
            "bundle",
            "import",
            "--bundle",
            str(bundle_path),
            "--profile",
            "imported-profile",
            "--format",
            "json",
        ]
    )
    assert code == 1

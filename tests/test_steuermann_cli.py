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
    _create_profile_dir(tmp_path)
    (tmp_path / ".env").write_text(
        "POSTGRES_PASSWORD=secret\nPROFILE_ID=starter\nLLM_PROVIDERS_LMSTUDIO_API_BASE=http://localhost:1234/v1\nEMBEDDING_SERVER=http://localhost:1234/v1\n",
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
    assert "LLM provider endpoints" in names
    assert "EMBEDDING_SERVER" in names


def test_config_explain_emits_source_chain(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROFILE_ID", "starter")
    monkeypatch.delenv("ACTIVE_PROFILE_ID", raising=False)

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



def test_config_contract_check_detects_prefix_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "config" / "contracts").mkdir(parents=True)
    bad_contract = {
        "schema_version": 1,
        "contract_name": "test",
        "precedence": ["base", "profile", "environment"],
        "sections": {
            "core": {"source_file": "config/core.yaml", "profile_overlay_file": "config/profiles/<profile_id>/core.yaml", "profile_mutability": "partial"},
            "features": {"source_file": "config/features.yaml", "profile_overlay_file": "config/profiles/<profile_id>/features.yaml", "profile_mutability": "partial"},
            "tools": {"profile_overlay_file": "config/profiles/<profile_id>/tools.yaml", "profile_mutability": "full"},
            "agents": {"profile_overlay_file": "config/profiles/<profile_id>/agents.yaml", "profile_mutability": "full"},
        },
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
        "sections": {
            "core": {"source_file": "config/core.yaml", "profile_overlay_file": "config/profiles/<profile_id>/core.yaml", "profile_mutability": "partial"},
            "features": {"source_file": "config/features.yaml", "profile_overlay_file": "config/profiles/<profile_id>/features.yaml", "profile_mutability": "partial"},
            "tools": {"profile_overlay_file": "config/profiles/<profile_id>/tools.yaml", "profile_mutability": "full"},
            "agents": {"profile_overlay_file": "config/profiles/<profile_id>/agents.yaml", "profile_mutability": "full"},
        },
        "policies": {"docs_mutation": "disabled", "manual_config_editing": "supported", "ingest_interface": "steuermann_ingest_only", "json_output_stability": "required"},
        "severity_policy": {"blocking": "error", "advisory": "warning"},
        "profile_safety": {
            "allowed_core_prefixes": ["profile.language", "profile.locale", "profile.timezone", "profile.supported_languages", "llm", "prompts", "tool_routing", "rag", "tokens", "memory.embeddings", "memory.retention"],
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
                "memory.embeddings",
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
                "memory.embeddings",
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
    monkeypatch.setattr(steuermann, "_iter_profiles", lambda: ["starter"])
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


def test_parse_env_value_strips_inline_comments_and_quotes() -> None:
    pv = steuermann._parse_env_value
    # double-quoted value with trailing inline comment
    assert pv('"Local Profile"  # Keep in sync') == "Local Profile"
    # single-quoted argon2 hash (contains $ signs)
    assert pv("'$argon2id$v=19$abc'") == "$argon2id$v=19$abc"
    # unquoted value with inline comment
    assert pv("lm-studio  # placeholder key") == "lm-studio"
    # unquoted value, no comment
    assert pv("http://host.docker.internal:1234/v1") == "http://host.docker.internal:1234/v1"
    # hash inside value without preceding space is preserved
    assert pv("pass#word") == "pass#word"
    # empty value
    assert pv("") == ""
    # tab before hash
    assert pv("value\t# comment") == "value"


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
                        "profile_overlay_file": "config/profiles/<profile_id>/tools.yaml",
                        "profile_mutability": "full",
                    },
                    "agents": {
                        "profile_overlay_file": "config/profiles/<profile_id>/agents.yaml",
                        "profile_mutability": "full",
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
                        "profile.language",
                        "profile.locale",
                        "profile.timezone",
                        "profile.supported_languages",
                        "llm",
                        "prompts",
                        "tool_routing",
                        "rag",
                        "tokens",
                        "memory.embeddings",
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


# ---------------------------------------------------------------------------
# setup init (first-time setup wizard)
# ---------------------------------------------------------------------------

import getpass  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402

import universal_agentic_framework.cli.setup_init as setup_init  # noqa: E402


_ENV_EXAMPLE = """\
# Steuermann env template
PROFILE_ID=starter
APP_UID=1000
APP_GID=1000
CHECKPOINTER_POSTGRES_DSN=postgresql://framework:framework@postgres:5432/framework
LLM_PROVIDERS_LMSTUDIO_API_BASE=http://host.docker.internal:1234/v1
LLM_PROVIDERS_OLLAMA_API_BASE=http://host.docker.internal:11434/v1
LLM_PROVIDERS_OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OPENAI_API_KEY=lm-studio
LLM_PROVIDERS_OPENROUTER_API_KEY=
EMBEDDING_SERVER=http://host.docker.internal:1234/v1
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=framework
POSTGRES_PASSWORD=
POSTGRES_DB=framework
AUTH_ENABLED=false
AUTH_USERNAME=admin
AUTH_ADMIN_EMAIL=admin@example.com
AUTH_PASSWORD_HASH=
AUTH_SESSION_SECRET=
CHAT_ACCESS_TOKEN=
"""

_STARTER_CORE = """\
llm:
  roles:
    chat:
      provider_id: lmstudio
      api_base: $LLM_PROVIDERS_LMSTUDIO_API_BASE
      model: openai/google/gemma-4-e4b
    vision:
      provider_id: lmstudio
      api_base: $LLM_PROVIDERS_LMSTUDIO_API_BASE
      model: openai/google/gemma-4-e4b
    auxiliary:
      provider_id: lmstudio
      api_base: $LLM_PROVIDERS_LMSTUDIO_API_BASE
      model: openai/google/gemma-4-e2b
    embedding:
      provider_id: lmstudio
      api_base: $LLM_PROVIDERS_LMSTUDIO_API_BASE
      model: openai/text-embedding-granite-embedding-278m-multilingual
memory:
  embeddings:
    dimension: 768
    remote_endpoint: $EMBEDDING_SERVER
  mem0:
    llm_provider: lmstudio
"""


def _write_env_example(tmp_path: Path) -> Path:
    path = tmp_path / ".env.example"
    path.write_text(_ENV_EXAMPLE, encoding="utf-8")
    return path


def _write_starter_core(profile_dir: Path) -> None:
    (profile_dir / "core.yaml").write_text(_STARTER_CORE, encoding="utf-8")


def _parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _stub_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real config loading + argon2 so the wizard tests stay light."""
    monkeypatch.setattr(
        steuermann, "_setup_check_payload", lambda env: {"status": "ok", "sections": []}
    )
    monkeypatch.setattr(setup_init, "generate_password_hash", lambda pw: "FAKEHASH")


def _set_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)


def _queue_inputs(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it, ""))
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: "interactive-admin-pw")


# 1. Non-TTY run writes a valid .env (starter, strong POSTGRES_PASSWORD).
# 2. Generated session secret + access token are 64-char hex.
# 9. CHECKPOINTER_POSTGRES_DSN password matches the generated POSTGRES_PASSWORD.
def test_setup_init_non_tty_writes_valid_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    _write_starter_core(profile_dir)
    _write_env_example(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)

    code = steuermann.main(["setup", "init", "--force", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "starter"
    assert payload["scaffolded_from"] is None

    env = _parse_env((tmp_path / ".env").read_text(encoding="utf-8"))
    assert env["PROFILE_ID"] == "starter"
    assert env["POSTGRES_PASSWORD"] and env["POSTGRES_PASSWORD"] != "framework"

    # 2: hex tokens
    for key in ("AUTH_SESSION_SECRET", "CHAT_ACCESS_TOKEN"):
        assert len(env[key]) == 64
        assert all(c in "0123456789abcdef" for c in env[key])

    # 9: DSN password matches
    assert f"framework:{env['POSTGRES_PASSWORD']}@" in env["CHECKPOINTER_POSTGRES_DSN"]


# 3. detect_platform() — arm64 + Pi5 flag; Windows (no os.getuid) → uid None.
def test_detect_platform_arm_and_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    plat = setup_init.detect_platform()
    assert plat["is_arm"] is True
    assert plat["pi5_qdrant_warning"] is True

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.delattr(os, "getgid", raising=False)
    plat2 = setup_init.detect_platform()
    assert plat2["uid"] is None
    assert plat2["gid"] is None
    assert plat2["is_arm"] is False


# 4. Existing .env backed up with --force; non-force non-TTY aborts untouched.
def test_setup_init_existing_env_backed_up_with_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    _write_starter_core(profile_dir)
    _write_env_example(tmp_path)
    (tmp_path / ".env").write_text("POSTGRES_PASSWORD=orig\nPROFILE_ID=starter\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)

    code = steuermann.main(["setup", "init", "--force", "--format", "json"])
    assert code == 0
    assert (tmp_path / ".env.bak").exists()
    assert "POSTGRES_PASSWORD=orig" in (tmp_path / ".env.bak").read_text(encoding="utf-8")


def test_setup_init_non_tty_without_force_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _create_profile_dir(tmp_path)
    _write_env_example(tmp_path)
    original = "POSTGRES_PASSWORD=orig\nPROFILE_ID=starter\n"
    (tmp_path / ".env").write_text(original, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)

    code = steuermann.main(["setup", "init", "--format", "json"])
    assert code == 2
    assert (tmp_path / ".env").read_text(encoding="utf-8") == original
    assert not (tmp_path / ".env.bak").exists()


# 5. --provider ollama writes the ollama endpoint var.
def test_setup_init_provider_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    _write_starter_core(profile_dir)
    _write_env_example(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)

    code = steuermann.main(["setup", "init", "--force", "--provider", "ollama", "--format", "json"])
    assert code == 0
    env = _parse_env((tmp_path / ".env").read_text(encoding="utf-8"))
    assert env["LLM_PROVIDERS_OLLAMA_API_BASE"] == "http://host.docker.internal:11434/v1"


# 5 (cont). --provider openrouter --openrouter-api-key writes the key.
def test_setup_init_provider_openrouter_writes_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    _write_starter_core(profile_dir)
    _write_env_example(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)

    code = steuermann.main(
        [
            "setup", "init", "--force",
            "--provider", "openrouter", "--openrouter-api-key", "sk-x",
            "--format", "json",
        ]
    )
    assert code == 0
    env = _parse_env((tmp_path / ".env").read_text(encoding="utf-8"))
    assert env["LLM_PROVIDERS_OPENROUTER_API_KEY"] == "sk-x"


# 6. Auth line is exactly AUTH_PASSWORD_HASH='<hash>' and AUTH_ENABLED=true.
def test_setup_init_auth_line_is_single_quoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    _write_starter_core(profile_dir)
    _write_env_example(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)

    code = steuermann.main(["setup", "init", "--force", "--format", "json"])
    assert code == 0
    env = _parse_env((tmp_path / ".env").read_text(encoding="utf-8"))
    assert env["AUTH_PASSWORD_HASH"] == "'FAKEHASH'"
    assert env["AUTH_ENABLED"] == "true"


# 7. Invalid --profile base aborts (exit 2) before any .env is written.
def test_setup_init_invalid_profile_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _create_profile_dir(tmp_path)
    _write_env_example(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)

    code = steuermann.main(["setup", "init", "--force", "--profile", "base", "--format", "json"])
    assert code == 2
    assert not (tmp_path / ".env").exists()


# 8. render_env_from_example preserves comments, changes only the value, appends extras.
def test_render_env_from_example_preserves_comments() -> None:
    example = "# a comment\nPROFILE_ID=starter\nFOO=bar\n"
    out, extras = setup_init.render_env_from_example(example, {"PROFILE_ID": "local"})
    assert "# a comment" in out
    assert "PROFILE_ID=local" in out
    assert "FOO=bar" in out
    assert extras == []

    out2, extras2 = setup_init.render_env_from_example("FOO=bar\n", {"NEW_KEY": "x"})
    assert "# Added by steuermann setup init" in out2
    assert "NEW_KEY=x" in out2
    assert extras2 == ["NEW_KEY"]


# 10. Simulated Linux: APP_UID/APP_GID equal the detected uid/gid.
def test_setup_init_linux_sets_app_uid_gid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    _write_starter_core(profile_dir)
    _write_env_example(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(os, "getuid", lambda: 4321, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 5678, raising=False)
    monkeypatch.setattr(os, "chown", lambda *a, **k: None, raising=False)

    code = steuermann.main(["setup", "init", "--force", "--format", "json"])
    assert code == 0
    env = _parse_env((tmp_path / ".env").read_text(encoding="utf-8"))
    assert env["APP_UID"] == "4321"
    assert env["APP_GID"] == "5678"


# 11. Re-run preserves an existing non-empty POSTGRES_PASSWORD.
def test_setup_init_preserves_existing_postgres_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    _write_starter_core(profile_dir)
    _write_env_example(tmp_path)
    (tmp_path / ".env").write_text(
        "POSTGRES_PASSWORD=preserved_secret_123\nPROFILE_ID=starter\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)

    code = steuermann.main(["setup", "init", "--force", "--format", "json"])
    assert code == 0
    env = _parse_env((tmp_path / ".env").read_text(encoding="utf-8"))
    assert env["POSTGRES_PASSWORD"] == "preserved_secret_123"
    assert "preserved_secret_123" in env["CHECKPOINTER_POSTGRES_DSN"]


# 12. write_profile_llm_config: openrouter fields + Mem0 provider; lmstudio keeps its branch.
def test_write_profile_llm_config_openrouter_and_lmstudio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = _create_profile_dir(tmp_path, "local")
    _write_starter_core(profile_dir)
    monkeypatch.chdir(tmp_path)

    def _role_cfg(provider: str, api_base: str, api_key: str, chat_model: str) -> dict:
        role = {"provider_id": provider, "api_base": api_base, "api_key": api_key}
        return {
            "chat": {**role, "model": chat_model},
            "vision": {**role, "model": chat_model},
            "auxiliary": {**role, "model": chat_model},
            "embedding": {"endpoint": "http://emb", "model": "openai/my-embed", "dimension": 1024},
        }

    written = setup_init.write_profile_llm_config(
        "local",
        _role_cfg(
            "openrouter",
            "$LLM_PROVIDERS_OPENROUTER_API_BASE",
            "$LLM_PROVIDERS_OPENROUTER_API_KEY",
            "openai/anthropic/claude-3.5-sonnet",
        ),
    )
    assert written is not None
    data = yaml.safe_load((profile_dir / "core.yaml").read_text(encoding="utf-8"))
    chat = data["llm"]["roles"]["chat"]
    assert chat["provider_id"] == "openrouter"
    assert chat["api_base"] == "$LLM_PROVIDERS_OPENROUTER_API_BASE"
    assert chat["api_key"] == "$LLM_PROVIDERS_OPENROUTER_API_KEY"
    assert chat["model"] == "openai/anthropic/claude-3.5-sonnet"
    assert data["memory"]["mem0"]["llm_provider"] == "openrouter"
    assert data["llm"]["roles"]["embedding"]["api_base"] == "$EMBEDDING_SERVER"
    assert data["llm"]["roles"]["embedding"]["model"] == "openai/my-embed"
    assert data["memory"]["embeddings"]["dimension"] == 1024
    assert not (profile_dir / "core.yaml.bak").exists()

    # lmstudio keeps mem0.llm_provider == lmstudio (the LM-Studio response-format branch).
    setup_init.write_profile_llm_config(
        "local",
        _role_cfg(
            "lmstudio", "$LLM_PROVIDERS_LMSTUDIO_API_BASE", "$OPENAI_API_KEY", "openai/google/gemma-4-e4b"
        ),
    )
    data2 = yaml.safe_load((profile_dir / "core.yaml").read_text(encoding="utf-8"))
    assert data2["memory"]["mem0"]["llm_provider"] == "lmstudio"


# 13. All-defaults run on starter leaves core.yaml untouched (no write).
def test_setup_init_all_defaults_leaves_core_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_dir = _create_profile_dir(tmp_path)
    _write_starter_core(profile_dir)
    before = (profile_dir / "core.yaml").read_bytes()
    _write_env_example(tmp_path)
    monkeypatch.chdir(tmp_path)
    _set_non_tty(monkeypatch)
    _stub_wizard(monkeypatch)

    code = steuermann.main(["setup", "init", "--force", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "starter"
    assert payload["profile_config_path"] is None
    assert (profile_dir / "core.yaml").read_bytes() == before
    env = _parse_env((tmp_path / ".env").read_text(encoding="utf-8"))
    assert env["PROFILE_ID"] == "starter"


# 14. Scaffold-on-customize: change chat model → scaffold local, starter unchanged.
def test_setup_init_scaffold_on_customize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    starter_dir = _create_profile_dir(tmp_path)
    _write_starter_core(starter_dir)
    starter_before = (starter_dir / "core.yaml").read_bytes()
    _write_env_example(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_wizard(monkeypatch)
    # provider, endpoint, key, chat(change), vision, aux, emb endpoint, emb model, emb dim, scaffold id
    _queue_inputs(
        monkeypatch,
        ["", "", "", "openai/acme/custom-chat", "", "", "", "", "", "local"],
    )

    code = steuermann.main(["setup", "init", "--force", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "local"
    assert payload["scaffolded_from"] == "starter"

    local_core = yaml.safe_load(
        (tmp_path / "config" / "profiles" / "local" / "core.yaml").read_text(encoding="utf-8")
    )
    assert local_core["llm"]["roles"]["chat"]["model"] == "openai/acme/custom-chat"
    env = _parse_env((tmp_path / ".env").read_text(encoding="utf-8"))
    assert env["PROFILE_ID"] == "local"
    # starter stays byte-for-byte pristine
    assert (starter_dir / "core.yaml").read_bytes() == starter_before


# 15. Custom embedding endpoint → .env EMBEDDING_SERVER set, profile references $EMBEDDING_SERVER.
def test_setup_init_embedding_endpoint_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    starter_dir = _create_profile_dir(tmp_path)
    _write_starter_core(starter_dir)
    _write_env_example(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_wizard(monkeypatch)
    # change only the embedding endpoint (triggers a profile rewrite via scaffold)
    _queue_inputs(
        monkeypatch,
        ["", "", "", "", "", "", "http://my-embed:9999/v1", "", "", "local"],
    )

    code = steuermann.main(["setup", "init", "--force", "--format", "json"])
    assert code == 0
    env = _parse_env((tmp_path / ".env").read_text(encoding="utf-8"))
    assert env["EMBEDDING_SERVER"] == "http://my-embed:9999/v1"

    local_core = yaml.safe_load(
        (tmp_path / "config" / "profiles" / "local" / "core.yaml").read_text(encoding="utf-8")
    )
    assert local_core["llm"]["roles"]["embedding"]["api_base"] == "$EMBEDDING_SERVER"
    assert local_core["memory"]["embeddings"]["remote_endpoint"] == "$EMBEDDING_SERVER"


# 16. _read_env_file_values correctly handles double-quoted values with inline comments.
def test_read_env_file_values_inline_comments(tmp_path: Path) -> None:
    env_text = (
        'SIMPLE=plain\n'
        'DOUBLE_QUOTED="Local Profile"  # Keep in sync\n'
        "SINGLE_QUOTED='$argon2id$v=19$abc'\n"
        'UNQUOTED=lm-studio  # placeholder\n'
        'HASH_IN_VALUE=pass#word\n'
    )
    path = tmp_path / ".env"
    path.write_text(env_text, encoding="utf-8")
    result = setup_init._read_env_file_values(path)
    assert result["SIMPLE"] == "plain"
    assert result["DOUBLE_QUOTED"] == "Local Profile"
    assert result["SINGLE_QUOTED"] == "$argon2id$v=19$abc"
    assert result["UNQUOTED"] == "lm-studio"
    assert result["HASH_IN_VALUE"] == "pass#word"

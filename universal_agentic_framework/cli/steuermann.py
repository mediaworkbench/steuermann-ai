"""Unified steuermann CLI for setup and configuration operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import tomllib
from typing import Any, Iterable

import yaml

from universal_agentic_framework.cli.ingest import add_ingest_subcommands
from universal_agentic_framework.config.loader import (
    _DISALLOWED_PROFILE_FEATURE_FLAGS,
    _PROFILE_CORE_ALLOWED_PREFIXES,
)
from universal_agentic_framework.config import (
    get_active_profile_id,
    get_profile_dir,
    load_agents_config,
    load_core_config,
    load_features_config,
    load_profile_metadata,
    load_profile_ui_config,
    load_tools_config,
)

PROFILE_REQUIRED_FILES = [
    "profile.yaml",
    "core.yaml",
    "features.yaml",
    "agents.yaml",
    "tools.yaml",
    "ui.yaml",
]
ENV_PLACEHOLDER_RE = re.compile(r"\$(\{?[A-Z0-9_]+\}?)")
CONFIG_SECTION_FILES = {
    "core": "core.yaml",
    "features": "features.yaml",
    "tools": "tools.yaml",
    "agents": "agents.yaml",
}
DOCS_CONFORMANCE_FILES = [
    Path("README.md"),
    Path("docs/index.md"),
    Path("docs/configuration.md"),
    Path("docs/profile_creation.md"),
]
CONTRACT_PATH = Path("config/contracts/cli_contract.yaml")
REQUIRED_CONTRACT_POLICIES = {
    "docs_mutation": "disabled",
    "manual_config_editing": "supported",
    "ingest_interface": "steuermann_ingest_only",
    "json_output_stability": "required",
}
EXPECTED_SECTION_MUTABILITY = {
    "core": "partial",
    "features": "partial",
    "tools": "partial",
    "agents": "partial",
}
EXPECTED_SEVERITY_POLICY = {
    "blocking": "error",
    "advisory": "warning",
}
EXPECTED_MUTATOR_SURFACE = {
    "config_set": {
        "profile_scope": "named_profile_only",
        "target_file": "core.yaml",
        "key_scope": "profile_safe_core_only",
        "dry_run_default": True,
        "requires_apply_flag": True,
        "requires_confirm_token": True,
        "confirm_token": "APPLY",
        "interactive_tty_prompt_fallback": True,
        "non_interactive_requires_confirm_flag": True,
        "rollback_on_validation_error": True,
    },
    "config_unset": {
        "profile_scope": "named_profile_only",
        "target_file": "core.yaml",
        "key_scope": "profile_safe_core_only",
        "dry_run_default": True,
        "requires_apply_flag": True,
        "requires_confirm_token": True,
        "confirm_token": "APPLY",
        "interactive_tty_prompt_fallback": True,
        "non_interactive_requires_confirm_flag": True,
        "rollback_on_validation_error": True,
    }
}
DEFAULT_BUNDLE_FRAMEWORK_VERSION_RANGE = ">=0.2,<1.0"
DEFAULT_BUNDLE_REQUIRED_KEYS = ["profile_id", "display_name", "description"]


def _framework_version() -> str:
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = str((data.get("tool") or {}).get("poetry", {}).get("version", "")).strip()
        if version:
            return version
    return "0.0.0"


def _parse_version_tuple(raw: str) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", raw)]
    if not parts:
        raise ValueError(f"Invalid version '{raw}'")
    return tuple(parts)


def _compare_versions(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    left_n = left + (0,) * (width - len(left))
    right_n = right + (0,) * (width - len(right))
    if left_n < right_n:
        return -1
    if left_n > right_n:
        return 1
    return 0


def _version_in_range(version: str, version_range: str) -> bool:
    ops = {
        ">": lambda cmp: cmp > 0,
        ">=": lambda cmp: cmp >= 0,
        "<": lambda cmp: cmp < 0,
        "<=": lambda cmp: cmp <= 0,
        "==": lambda cmp: cmp == 0,
    }
    version_tuple = _parse_version_tuple(version)
    clauses = [item.strip() for item in version_range.split(",") if item.strip()]
    if not clauses:
        return False
    for clause in clauses:
        match = re.match(r"^(>=|<=|>|<|==)\s*([0-9][0-9A-Za-z._-]*)$", clause)
        if not match:
            return False
        op, raw_target = match.groups()
        target_tuple = _parse_version_tuple(raw_target)
        compare_result = _compare_versions(version_tuple, target_tuple)
        if not ops[op](compare_result):
            return False
    return True


def _profile_compatibility(profile_dir: Path, profile_id: str) -> dict[str, Any]:
    compatibility: dict[str, Any] = {
        "framework_version": _framework_version(),
        "framework_version_range": DEFAULT_BUNDLE_FRAMEWORK_VERSION_RANGE,
        "minimum_required_keys": list(DEFAULT_BUNDLE_REQUIRED_KEYS),
    }

    profile_yaml = profile_dir / "profile.yaml"
    if profile_yaml.exists():
        profile_data = yaml.safe_load(profile_yaml.read_text(encoding="utf-8")) or {}
        configured = profile_data.get("compatibility")
        if isinstance(configured, dict):
            range_value = configured.get("framework_version_range")
            if isinstance(range_value, str) and range_value.strip():
                compatibility["framework_version_range"] = range_value.strip()
            keys_value = configured.get("minimum_required_keys")
            if isinstance(keys_value, list) and all(isinstance(item, str) for item in keys_value):
                compatibility["minimum_required_keys"] = list(keys_value)

    compatibility["profile_id"] = profile_id
    return compatibility


def _with_profile_env(profile_id: str | None) -> dict[str, str]:
    env = _load_env_file()
    if profile_id:
        env["PROFILE_ID"] = profile_id
    return env


def _stable_dump(payload: Any, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(payload, indent=2, sort_keys=True)
    return yaml.safe_dump(payload, sort_keys=True, allow_unicode=False)


def _print_payload(payload: Any, fmt: str) -> None:
    print(_stable_dump(payload, fmt))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_env_file(path: Path = Path(".env")) -> dict[str, str]:
    env = dict(os.environ)
    if not path.exists():
        return env

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env.setdefault(key, value)
    return env


def _collect_unresolved_placeholders(value: Any, path: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            found.extend(_collect_unresolved_placeholders(child, next_path))
        return found
    if isinstance(value, list):
        for idx, child in enumerate(value):
            next_path = f"{path}[{idx}]"
            found.extend(_collect_unresolved_placeholders(child, next_path))
        return found
    if isinstance(value, str) and ENV_PLACEHOLDER_RE.search(value):
        found.append({"path": path or "root", "value": value})
    return found


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if isinstance(model, dict):
        return model
    raise TypeError("Unsupported model payload")


def _load_effective_configs(profile_id: str | None = None) -> dict[str, Any]:
    env = _with_profile_env(profile_id)
    core = load_core_config(env=env)
    features = load_features_config(env=env)
    tools = load_tools_config(env=env)
    agents = load_agents_config(env=env)

    resolved_profile = get_active_profile_id(env)
    payload: dict[str, Any] = {
        "active_profile": resolved_profile,
        "core": _model_to_dict(core),
        "features": _model_to_dict(features),
        "tools": _model_to_dict(tools),
        "agents": _model_to_dict(agents),
    }

    if resolved_profile != "base":
        payload["profile_metadata"] = _model_to_dict(load_profile_metadata(env=env))
        payload["ui"] = _model_to_dict(load_profile_ui_config(env=env))
    else:
        payload["profile_metadata"] = None
        payload["ui"] = _model_to_dict(load_profile_ui_config(env=env))

    return payload


def _dig(data: dict[str, Any], dot_key: str) -> Any:
    current: Any = data
    for part in dot_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(dot_key)
    return current


def _dig_optional(data: dict[str, Any], dot_key: str) -> tuple[bool, Any]:
    try:
        return True, _dig(data, dot_key)
    except KeyError:
        return False, None


def _set_dot_path(data: dict[str, Any], dot_key: str, value: Any) -> None:
    parts = dot_key.split(".")
    current: dict[str, Any] = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if next_value is None:
            current[part] = {}
            next_value = current[part]
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested key under non-object path: {part}")
        current = next_value
    current[parts[-1]] = value


def _unset_dot_path(data: dict[str, Any], dot_key: str) -> tuple[bool, Any]:
    parts = dot_key.split(".")
    current: Any = data
    lineage: list[tuple[dict[str, Any], str]] = []

    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return False, None
        lineage.append((current, part))
        current = current[part]

    last = parts[-1]
    if not isinstance(current, dict) or last not in current:
        return False, None

    old_value = current[last]
    del current[last]

    for parent, key in reversed(lineage):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            del parent[key]
        else:
            break

    return True, old_value

def _is_profile_safe_core_key(key: str) -> bool:
    if not key.startswith("core."):
        return False
    core_subkey = key.split(".", 1)[1]
    return any(
        core_subkey == allowed or core_subkey.startswith(f"{allowed}.")
        for allowed in _PROFILE_CORE_ALLOWED_PREFIXES
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _section_paths(section: str, active_profile: str) -> tuple[Path | None, Path | None]:
    if section in CONFIG_SECTION_FILES:
        filename = CONFIG_SECTION_FILES[section]
        base_path = Path("config") / filename
        profile_path = Path("config/profiles") / active_profile / filename if active_profile != "base" else None
        return base_path, profile_path

    if section == "ui":
        return None, (Path("config/profiles") / active_profile / "ui.yaml" if active_profile != "base" else None)

    if section == "profile_metadata":
        return None, (Path("config/profiles") / active_profile / "profile.yaml" if active_profile != "base" else None)

    return None, None


def _env_refs(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    refs: list[str] = []
    for raw in ENV_PLACEHOLDER_RE.findall(value):
        refs.append(raw.strip("{}"))
    return refs


def _contract_parity_report(contract: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected_sections = set(CONFIG_SECTION_FILES)
    contract_sections = set((contract.get("sections") or {}).keys())

    missing_sections = sorted(expected_sections - contract_sections)
    extra_sections = sorted(contract_sections - expected_sections)
    checks.append(
        {
            "path": str(CONTRACT_PATH),
            "status": "ok" if not missing_sections else "fail",
            "details": (
                "Contract covers expected CLI/runtime config sections"
                if not missing_sections
                else f"Missing contract sections: {', '.join(missing_sections)}"
            ),
        }
    )
    checks.append(
        {
            "path": str(CONTRACT_PATH),
            "status": "ok" if not extra_sections else "fail",
            "details": (
                "No unknown contract sections"
                if not extra_sections
                else f"Unknown contract sections: {', '.join(extra_sections)}"
            ),
        }
    )

    for section, filename in CONFIG_SECTION_FILES.items():
        section_data = (contract.get("sections") or {}).get(section, {})
        expected_source = f"config/{filename}"
        actual_source = section_data.get("source_file")
        expected_overlay = f"config/profiles/<profile_id>/{filename}"
        actual_overlay = section_data.get("profile_overlay_file")
        expected_mutability = EXPECTED_SECTION_MUTABILITY[section]
        actual_mutability = section_data.get("profile_mutability")
        checks.append(
            {
                "path": str(CONTRACT_PATH),
                "status": "ok" if actual_source == expected_source else "fail",
                "details": (
                    f"{section}.source_file matches {expected_source}"
                    if actual_source == expected_source
                    else f"{section}.source_file mismatch: expected {expected_source}, got {actual_source}"
                ),
            }
        )
        checks.append(
            {
                "path": str(CONTRACT_PATH),
                "status": "ok" if actual_overlay == expected_overlay else "fail",
                "details": (
                    f"{section}.profile_overlay_file matches {expected_overlay}"
                    if actual_overlay == expected_overlay
                    else f"{section}.profile_overlay_file mismatch: expected {expected_overlay}, got {actual_overlay}"
                ),
            }
        )
        checks.append(
            {
                "path": str(CONTRACT_PATH),
                "status": "ok" if actual_mutability == expected_mutability else "fail",
                "details": (
                    f"{section}.profile_mutability is {expected_mutability}"
                    if actual_mutability == expected_mutability
                    else f"{section}.profile_mutability mismatch: expected {expected_mutability}, got {actual_mutability}"
                ),
            }
        )

    precedence = contract.get("precedence") or []
    expected_precedence = ["base", "profile", "environment"]
    checks.append(
        {
            "path": str(CONTRACT_PATH),
            "status": "ok" if precedence == expected_precedence else "fail",
            "details": (
                "Precedence matches base -> profile -> environment"
                if precedence == expected_precedence
                else f"Precedence mismatch: expected {expected_precedence}, got {precedence}"
            ),
        }
    )

    contract_policies = contract.get("policies") or {}
    for policy_name, expected_value in REQUIRED_CONTRACT_POLICIES.items():
        actual_value = contract_policies.get(policy_name)
        checks.append(
            {
                "path": str(CONTRACT_PATH),
                "status": "ok" if actual_value == expected_value else "fail",
                "details": (
                    f"{policy_name} policy is {expected_value}"
                    if actual_value == expected_value
                    else f"{policy_name} policy mismatch: expected {expected_value}, got {actual_value}"
                ),
            }
        )

    severity_policy = contract.get("severity_policy") or {}
    for key, expected_value in EXPECTED_SEVERITY_POLICY.items():
        actual_value = severity_policy.get(key)
        checks.append(
            {
                "path": str(CONTRACT_PATH),
                "status": "ok" if actual_value == expected_value else "fail",
                "details": (
                    f"severity_policy.{key} is {expected_value}"
                    if actual_value == expected_value
                    else f"severity_policy.{key} mismatch: expected {expected_value}, got {actual_value}"
                ),
            }
        )

    # --- schema-field coverage: profile_safety ---
    profile_safety = contract.get("profile_safety") or {}
    contract_prefixes = set(profile_safety.get("allowed_core_prefixes") or [])
    runtime_prefixes = _PROFILE_CORE_ALLOWED_PREFIXES
    missing_prefixes = sorted(runtime_prefixes - contract_prefixes)
    extra_prefixes = sorted(contract_prefixes - runtime_prefixes)
    if not missing_prefixes and not extra_prefixes:
        prefix_detail = "profile_safety.allowed_core_prefixes matches loader constants"
    else:
        parts = []
        if missing_prefixes:
            parts.append(f"missing: {', '.join(missing_prefixes)}")
        if extra_prefixes:
            parts.append(f"extra: {', '.join(extra_prefixes)}")
        prefix_detail = f"profile_safety.allowed_core_prefixes drift — {'; '.join(parts)}"
    checks.append(
        {
            "path": str(CONTRACT_PATH),
            "status": "ok" if not missing_prefixes and not extra_prefixes else "fail",
            "details": prefix_detail,
        }
    )

    contract_flags = set(profile_safety.get("disallowed_feature_flags") or [])
    runtime_flags = _DISALLOWED_PROFILE_FEATURE_FLAGS
    missing_flags = sorted(runtime_flags - contract_flags)
    extra_flags = sorted(contract_flags - runtime_flags)
    if not missing_flags and not extra_flags:
        flag_detail = "profile_safety.disallowed_feature_flags matches loader constants"
    else:
        parts = []
        if missing_flags:
            parts.append(f"missing: {', '.join(missing_flags)}")
        if extra_flags:
            parts.append(f"extra: {', '.join(extra_flags)}")
        flag_detail = f"profile_safety.disallowed_feature_flags drift — {'; '.join(parts)}"
    checks.append(
        {
            "path": str(CONTRACT_PATH),
            "status": "ok" if not missing_flags and not extra_flags else "fail",
            "details": flag_detail,
        }
    )

    profile_bundle_compatibility = contract.get("profile_bundle_compatibility") or {}
    contract_default_range = profile_bundle_compatibility.get("framework_version_range_default")
    checks.append(
        {
            "path": str(CONTRACT_PATH),
            "status": "ok" if contract_default_range == DEFAULT_BUNDLE_FRAMEWORK_VERSION_RANGE else "fail",
            "details": (
                "profile_bundle_compatibility.framework_version_range_default matches CLI default"
                if contract_default_range == DEFAULT_BUNDLE_FRAMEWORK_VERSION_RANGE
                else (
                    "profile_bundle_compatibility.framework_version_range_default mismatch: "
                    f"expected {DEFAULT_BUNDLE_FRAMEWORK_VERSION_RANGE}, got {contract_default_range}"
                )
            ),
        }
    )

    contract_required_keys = set(profile_bundle_compatibility.get("minimum_required_keys_default") or [])
    runtime_required_keys = set(DEFAULT_BUNDLE_REQUIRED_KEYS)
    missing_required_keys = sorted(runtime_required_keys - contract_required_keys)
    extra_required_keys = sorted(contract_required_keys - runtime_required_keys)
    if not missing_required_keys and not extra_required_keys:
        required_keys_detail = "profile_bundle_compatibility.minimum_required_keys_default matches CLI defaults"
    else:
        parts = []
        if missing_required_keys:
            parts.append(f"missing: {', '.join(missing_required_keys)}")
        if extra_required_keys:
            parts.append(f"extra: {', '.join(extra_required_keys)}")
        required_keys_detail = (
            "profile_bundle_compatibility.minimum_required_keys_default drift — "
            f"{'; '.join(parts)}"
        )
    checks.append(
        {
            "path": str(CONTRACT_PATH),
            "status": "ok" if not missing_required_keys and not extra_required_keys else "fail",
            "details": required_keys_detail,
        }
    )

    mutator_surface = contract.get("mutator_surface") or {}
    for command_name, expected_config in EXPECTED_MUTATOR_SURFACE.items():
        contract_config = mutator_surface.get(command_name) or {}
        for key, expected_value in expected_config.items():
            actual_value = contract_config.get(key)
            checks.append(
                {
                    "path": str(CONTRACT_PATH),
                    "status": "ok" if actual_value == expected_value else "fail",
                    "details": (
                        f"mutator_surface.{command_name}.{key} is {expected_value}"
                        if actual_value == expected_value
                        else (
                            f"mutator_surface.{command_name}.{key} mismatch: "
                            f"expected {expected_value}, got {actual_value}"
                        )
                    ),
                }
            )

    return checks


def _contract_check_payload() -> dict[str, Any]:
    contract = _read_yaml(CONTRACT_PATH)
    checks: list[dict[str, Any]] = [
        {
            "path": str(CONTRACT_PATH),
            "status": "ok" if bool(contract) else "fail",
            "details": "Contract loaded" if contract else "Contract file missing or invalid",
        }
    ]
    if contract:
        schema_ok = contract.get("schema_version") == 1
        checks.append(
            {
                "path": str(CONTRACT_PATH),
                "status": "ok" if schema_ok else "fail",
                "details": "schema_version is 1" if schema_ok else "schema_version mismatch",
            }
        )
        checks.extend(_contract_parity_report(contract))

    has_fail = any(check["status"] == "fail" for check in checks)
    return {"status": "error" if has_fail else "ok", "checks": checks}


def _profiles_root() -> Path:
    return Path("config/profiles")


def _iter_profiles() -> list[str]:
    root = _profiles_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def cmd_profile_active(args: argparse.Namespace) -> int:
    env = _with_profile_env(args.profile)
    profile_id = get_active_profile_id(env)
    profile_dir = get_profile_dir(profile_id=profile_id, env=env)
    metadata_valid = True
    error = None
    if profile_id != "base":
        try:
            load_profile_metadata(env=env)
        except Exception as exc:
            metadata_valid = False
            error = str(exc)

    payload = {
        "active_profile": profile_id,
        "profile_dir": str(profile_dir) if profile_dir else None,
        "metadata_valid": metadata_valid,
        "error": error,
    }
    _print_payload(payload, args.format)
    return 0 if metadata_valid else 1


def cmd_config_show(args: argparse.Namespace) -> int:
    payload = _load_effective_configs(args.profile)
    if args.section:
        if args.section not in payload:
            print(f"Unknown section: {args.section}", file=sys.stderr)
            return 1
        payload = {args.section: payload[args.section], "active_profile": payload["active_profile"]}
    _print_payload(payload, args.format)
    return 0


def cmd_config_explain(args: argparse.Namespace) -> int:
    env = _with_profile_env(args.profile)
    active = get_active_profile_id(env)
    effective = _load_effective_configs(args.profile)

    try:
        value = _dig(effective, args.key)
    except KeyError:
        print(f"Key not found: {args.key}", file=sys.stderr)
        return 1

    key_parts = args.key.split(".", 1)
    section = key_parts[0]
    subkey = key_parts[1] if len(key_parts) > 1 else ""
    base_path, profile_path = _section_paths(section, active)

    source_chain: list[dict[str, Any]] = []
    merged_raw: dict[str, Any] = {}

    if base_path:
        base_data = _read_yaml(base_path)
        merged_raw = _deep_merge(merged_raw, base_data)
        exists, _ = _dig_optional(base_data, subkey) if subkey else (True, base_data)
        source_chain.append({
            "layer": "base",
            "source": str(base_path),
            "key_present": exists,
        })

    if profile_path:
        profile_data = _read_yaml(profile_path)
        if profile_data:
            merged_raw = _deep_merge(merged_raw, profile_data)
        exists, _ = _dig_optional(profile_data, subkey) if subkey else (bool(profile_data), profile_data)
        source_chain.append({
            "layer": "profile",
            "source": str(profile_path),
            "key_present": exists,
        })

    raw_exists, raw_value = _dig_optional(merged_raw, subkey) if subkey else (bool(merged_raw), merged_raw)
    refs = _env_refs(raw_value)
    if refs:
        env_state = {name: ("set" if name in env else "missing") for name in refs}
        source_chain.append(
            {
                "layer": "environment",
                "source": "process environment",
                "references": refs,
                "reference_state": env_state,
            }
        )
    else:
        source_chain.append({"layer": "environment", "source": "process environment", "references": []})

    payload = {
        "active_profile": active,
        "key": args.key,
        "value": value,
        "raw_value_pre_env": raw_value if raw_exists else None,
        "source_chain": source_chain,
    }
    _print_payload(payload, args.format)
    return 0


def _validate_profile_files(profile_dir: Path) -> list[str]:
    missing: list[str] = []
    for name in PROFILE_REQUIRED_FILES:
        if not (profile_dir / name).exists():
            missing.append(name)
    prompts_dir = profile_dir / "prompts"
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        missing.append("prompts/")
    return missing


def _validate_one_profile(profile_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "profile": profile_id,
        "status": "ok",
        "errors": [],
        "warnings": [],
    }

    env = _with_profile_env(profile_id)
    profile_dir = get_profile_dir(profile_id=profile_id, env=env, require_exists=False)
    if profile_id != "base" and profile_dir:
        missing = _validate_profile_files(profile_dir)
        if missing:
            result["status"] = "error"
            result["errors"].append(f"Missing required files: {', '.join(missing)}")

    try:
        effective = _load_effective_configs(profile_id)
        unresolved = _collect_unresolved_placeholders(effective)
        if unresolved:
            result["warnings"].append(f"Unresolved placeholders: {len(unresolved)}")
            result["unresolved_placeholders"] = unresolved
    except Exception as exc:
        result["status"] = "error"
        result["errors"].append(str(exc))

    return result


def cmd_config_validate(args: argparse.Namespace) -> int:
    profiles = [args.profile] if args.profile else ["base"] + _iter_profiles()
    results = [_validate_one_profile(profile) for profile in profiles]
    has_error = any(item["status"] == "error" for item in results)
    has_warning = any(item.get("warnings") for item in results)
    payload = {
        "results": results,
        "status": "error" if has_error else ("warning" if has_warning else "ok"),
    }
    _print_payload(payload, args.format)
    return 1 if has_error or (args.strict and has_warning) else 0


def _resolve_apply_confirmation(confirm_value: str, payload: dict[str, Any]) -> tuple[bool, str | None]:
    if confirm_value == "APPLY":
        return True, "flag"

    if not sys.stdin.isatty():
        payload["error"] = "Apply mode requires --confirm APPLY"
        return False, None

    entered = input("Type APPLY to persist this change: ").strip()
    if entered == "APPLY":
        return True, "interactive"

    payload["error"] = "Apply confirmation failed; expected APPLY"
    return False, None

def cmd_config_set(args: argparse.Namespace) -> int:
    profile_id = args.profile
    payload: dict[str, Any] = {
        "status": "error",
        "profile": profile_id,
        "key": args.key,
        "apply": bool(args.apply),
    }

    if profile_id == "base":
        payload["error"] = "Mutating base profile is not supported; target a named profile overlay"
        _print_payload(payload, args.format)
        return 1

    if not _is_profile_safe_core_key(args.key):
        payload["error"] = "Only profile-safe core keys can be set"
        payload["allowed_core_prefixes"] = sorted(_PROFILE_CORE_ALLOWED_PREFIXES)
        _print_payload(payload, args.format)
        return 1

    profile_dir = _profiles_root() / profile_id
    core_path = profile_dir / "core.yaml"
    if not core_path.exists():
        payload["error"] = f"Profile core file missing: {core_path}"
        _print_payload(payload, args.format)
        return 1

    try:
        parsed_value = yaml.safe_load(args.value)
    except Exception as exc:
        payload["error"] = f"Failed to parse --value as YAML: {exc}"
        _print_payload(payload, args.format)
        return 1

    current_data = _read_yaml(core_path)
    profile_key = args.key.split(".", 1)[1]
    old_exists, old_value = _dig_optional(current_data, profile_key)
    updated_data = _deep_merge({}, current_data)
    _set_dot_path(updated_data, profile_key, parsed_value)

    payload["file"] = str(core_path)
    payload["old_value"] = old_value if old_exists else None
    payload["new_value"] = parsed_value
    payload["would_change"] = (old_value != parsed_value) if old_exists else True

    if not args.apply:
        payload["status"] = "ok"
        payload["mode"] = "dry-run"
        _print_payload(payload, args.format)
        return 0

    confirmed, confirmation_source = _resolve_apply_confirmation(args.confirm, payload)
    if not confirmed:
        payload["mode"] = "apply"
        _print_payload(payload, args.format)
        return 1
    payload["confirmation_source"] = confirmation_source

    original_text = core_path.read_text(encoding="utf-8")
    core_path.write_text(
        yaml.safe_dump(updated_data, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )

    validation = _validate_one_profile(profile_id)
    payload["validation"] = validation
    if validation.get("status") == "error":
        core_path.write_text(original_text, encoding="utf-8")
        payload["status"] = "error"
        payload["mode"] = "apply"
        payload["reverted"] = True
        payload["error"] = "Applied change failed profile validation and was reverted"
        _print_payload(payload, args.format)
        return 1

    payload["status"] = "ok"
    payload["mode"] = "apply"
    payload["reverted"] = False
    _print_payload(payload, args.format)
    return 0


def cmd_config_unset(args: argparse.Namespace) -> int:
    profile_id = args.profile
    payload: dict[str, Any] = {
        "status": "error",
        "profile": profile_id,
        "key": args.key,
        "apply": bool(args.apply),
    }

    if profile_id == "base":
        payload["error"] = "Mutating base profile is not supported; target a named profile overlay"
        _print_payload(payload, args.format)
        return 1

    if not _is_profile_safe_core_key(args.key):
        payload["error"] = "Only profile-safe core keys can be unset"
        payload["allowed_core_prefixes"] = sorted(_PROFILE_CORE_ALLOWED_PREFIXES)
        _print_payload(payload, args.format)
        return 1

    profile_dir = _profiles_root() / profile_id
    core_path = profile_dir / "core.yaml"
    if not core_path.exists():
        payload["error"] = f"Profile core file missing: {core_path}"
        _print_payload(payload, args.format)
        return 1

    current_data = _read_yaml(core_path)
    profile_key = args.key.split(".", 1)[1]
    updated_data = _deep_merge({}, current_data)
    found, old_value = _unset_dot_path(updated_data, profile_key)

    payload["file"] = str(core_path)
    payload["old_value"] = old_value if found else None
    payload["new_value"] = None
    payload["would_change"] = found

    if not args.apply:
        payload["status"] = "ok"
        payload["mode"] = "dry-run"
        _print_payload(payload, args.format)
        return 0

    confirmed, confirmation_source = _resolve_apply_confirmation(args.confirm, payload)
    if not confirmed:
        payload["mode"] = "apply"
        _print_payload(payload, args.format)
        return 1
    payload["confirmation_source"] = confirmation_source

    if not found:
        payload["status"] = "ok"
        payload["mode"] = "apply"
        payload["reverted"] = False
        payload["note"] = "Key not present; no changes applied"
        _print_payload(payload, args.format)
        return 0

    original_text = core_path.read_text(encoding="utf-8")
    core_path.write_text(
        yaml.safe_dump(updated_data, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )

    validation = _validate_one_profile(profile_id)
    payload["validation"] = validation
    if validation.get("status") == "error":
        core_path.write_text(original_text, encoding="utf-8")
        payload["status"] = "error"
        payload["mode"] = "apply"
        payload["reverted"] = True
        payload["error"] = "Applied change failed profile validation and was reverted"
        _print_payload(payload, args.format)
        return 1

    payload["status"] = "ok"
    payload["mode"] = "apply"
    payload["reverted"] = False
    _print_payload(payload, args.format)
    return 0


def cmd_config_contract_check(args: argparse.Namespace) -> int:
    payload = _contract_check_payload()
    _print_payload(payload, args.format)
    return 1 if payload["status"] == "error" else 0


def cmd_setup_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    env = _load_env_file()

    def add_check(name: str, ok: bool, blocking: bool, details: str) -> None:
        checks.append({
            "name": name,
            "status": "ok" if ok else "fail",
            "blocking": blocking,
            "details": details,
        })

    env_file = Path(".env")
    add_check(
        ".env presence",
        env_file.exists(),
        False,
        "Found local .env file" if env_file.exists() else "No .env file found; relying on process environment only",
    )

    postgres_password = env.get("POSTGRES_PASSWORD")
    add_check(
        "POSTGRES_PASSWORD",
        bool(postgres_password),
        True,
        "Set POSTGRES_PASSWORD in .env or environment" if not postgres_password else "Present",
    )

    llm_provider_vars = {
        "LLM_PROVIDERS_LMSTUDIO_API_BASE": env.get("LLM_PROVIDERS_LMSTUDIO_API_BASE"),
        "LLM_PROVIDERS_OLLAMA_API_BASE": env.get("LLM_PROVIDERS_OLLAMA_API_BASE"),
        "LLM_PROVIDERS_OPENROUTER_API_BASE": env.get("LLM_PROVIDERS_OPENROUTER_API_BASE"),
    }
    configured_providers = {k: v for k, v in llm_provider_vars.items() if v}
    add_check(
        "LLM provider endpoints",
        bool(configured_providers),
        False,
        (
            "Set at least one of LLM_PROVIDERS_LMSTUDIO_API_BASE, LLM_PROVIDERS_OLLAMA_API_BASE, "
            "LLM_PROVIDERS_OPENROUTER_API_BASE"
        )
        if not configured_providers
        else ", ".join(f"{k}={v}" for k, v in configured_providers.items()),
    )

    embedding_server = env.get("EMBEDDING_SERVER")
    add_check(
        "EMBEDDING_SERVER",
        bool(embedding_server),
        False,
        "Set EMBEDDING_SERVER for embedding-backed memory and ingestion" if not embedding_server else embedding_server,
    )

    profile_id = get_active_profile_id(env)
    profile_dir = get_profile_dir(profile_id=profile_id, env=env, require_exists=False)
    profile_ok = profile_id == "base" or (profile_dir is not None and profile_dir.exists())
    add_check(
        "PROFILE_ID",
        profile_ok,
        True,
        "Profile directory missing" if not profile_ok else f"Resolved profile: {profile_id}",
    )

    ingest_collection = env.get("INGEST_COLLECTION")
    try:
        core = load_core_config(env=env)
        rag_collection = core.rag.collection_name
        aligned = not ingest_collection or ingest_collection == rag_collection
        add_check(
            "INGEST_COLLECTION alignment",
            aligned,
            False,
            (
                "INGEST_COLLECTION differs from core.rag.collection_name"
                if not aligned
                else f"Aligned collection: {rag_collection}"
            ),
        )
    except Exception as exc:
        add_check("Core config load", False, True, f"Failed to load core config: {exc}")

    if args.probe_endpoints:
        import httpx

        if not configured_providers:
            add_check("LLM endpoint probe", False, False, "No LLM provider endpoints configured")
        else:
            for var_name, endpoint in configured_providers.items():
                probe_ok = False
                probe_details = f"{var_name} missing"
                try:
                    with httpx.Client(timeout=3.0) as client:
                        response = client.get(endpoint)
                    probe_ok = response.status_code < 500
                    probe_details = f"HTTP {response.status_code}"
                except Exception as exc:
                    probe_ok = False
                    probe_details = str(exc)
                add_check(f"LLM endpoint probe ({var_name})", probe_ok, False, probe_details)

    has_blocking_fail = any(c["status"] == "fail" and c["blocking"] for c in checks)
    payload = {
        "status": "error" if has_blocking_fail else "ok",
        "checks": checks,
    }
    _print_payload(payload, args.format)
    return 1 if has_blocking_fail else 0


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copytree(src, dst)


def cmd_profile_scaffold(args: argparse.Namespace) -> int:
    starter_dir = _profiles_root() / args.from_profile
    if not starter_dir.exists():
        print(f"Starter profile not found: {starter_dir}", file=sys.stderr)
        return 1

    target = _profiles_root() / args.profile
    _copytree(starter_dir, target)

    profile_id = args.profile
    profile_yaml = target / "profile.yaml"
    if profile_yaml.exists():
        data = yaml.safe_load(profile_yaml.read_text(encoding="utf-8")) or {}
        data["profile_id"] = profile_id
        profile_yaml.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=False), encoding="utf-8")

    # Bundle metadata goes in bundle_manifest.yaml, not profile.yaml

    manifest = {
        "schema_version": 1,
        "manifest_version": 1,
        "profile_id": profile_id,
        "compatibility": _profile_compatibility(target, profile_id),
        "required_files": PROFILE_REQUIRED_FILES + ["prompts/"],
    }
    (target / "bundle_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )

    _print_payload({"status": "ok", "target": str(target), "profile_id": profile_id}, args.format)
    return 0


def _bundle_manifest(profile_id: str, profile_dir: Path | None = None) -> dict[str, Any]:
    resolved_profile_dir = profile_dir or (_profiles_root() / profile_id)
    return {
        "schema_version": 1,
        "manifest_version": 1,
        "profile_id": profile_id,
        "compatibility": _profile_compatibility(resolved_profile_dir, profile_id),
        "required_files": PROFILE_REQUIRED_FILES + ["prompts/"],
    }


def cmd_profile_bundle_export(args: argparse.Namespace) -> int:
    profile_dir = _profiles_root() / args.profile
    if not profile_dir.exists():
        print(f"Profile not found: {profile_dir}", file=sys.stderr)
        return 1

    missing = _validate_profile_files(profile_dir)
    if missing:
        print(f"Profile missing required files: {', '.join(missing)}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(profile_dir, arcname=args.profile)
        manifest_bytes = yaml.safe_dump(
            _bundle_manifest(args.profile, profile_dir=profile_dir), sort_keys=True, allow_unicode=False
        ).encode("utf-8")
        info = tarfile.TarInfo(name="bundle_manifest.yaml")
        info.size = len(manifest_bytes)
        tf.addfile(info, fileobj=_BytesReader(manifest_bytes))

    _print_payload({"status": "ok", "bundle": str(out_path), "profile": args.profile}, args.format)
    return 0


class _BytesReader:
    def __init__(self, data: bytes):
        self._data = data
        self._idx = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._data) - self._idx
        chunk = self._data[self._idx : self._idx + size]
        self._idx += len(chunk)
        return chunk


def cmd_profile_bundle_import(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle)
    if not bundle.exists():
        print(f"Bundle not found: {bundle}", file=sys.stderr)
        return 1

    target = _profiles_root() / args.profile
    if target.exists():
        print(f"Target already exists: {target}", file=sys.stderr)
        return 1

    with tarfile.open(bundle, "r:gz") as tf:
        names = tf.getnames()
        if "bundle_manifest.yaml" not in names:
            print("Bundle manifest missing", file=sys.stderr)
            return 1
        manifest_file = tf.extractfile("bundle_manifest.yaml")
        assert manifest_file is not None
        manifest = yaml.safe_load(manifest_file.read().decode("utf-8")) or {}
        if manifest.get("schema_version") != 1:
            print("Unsupported bundle schema version", file=sys.stderr)
            return 1

        compatibility = manifest.get("compatibility") or {}
        if not isinstance(compatibility, dict):
            print("Bundle compatibility metadata missing or invalid", file=sys.stderr)
            return 1

        framework_version_range = compatibility.get("framework_version_range")
        if not isinstance(framework_version_range, str) or not framework_version_range.strip():
            print("Bundle compatibility.framework_version_range missing", file=sys.stderr)
            return 1

        current_version = _framework_version()
        if not _version_in_range(current_version, framework_version_range):
            print(
                (
                    "Bundle incompatible with current framework version "
                    f"{current_version}; required range: {framework_version_range}"
                ),
                file=sys.stderr,
            )
            return 1

        required_keys = compatibility.get("minimum_required_keys") or []
        if not isinstance(required_keys, list) or not all(isinstance(item, str) for item in required_keys):
            print("Bundle compatibility.minimum_required_keys must be a string list", file=sys.stderr)
            return 1

        members = [n for n in names if n != "bundle_manifest.yaml"]
        root_dirs = sorted({n.split("/")[0] for n in members if "/" in n})
        if not root_dirs:
            print("Invalid bundle content", file=sys.stderr)
            return 1

        profile_root = root_dirs[0]
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)

        # Extract into an isolated staging directory first so existing
        # profiles in parent are never modified by tar extraction.
        with tempfile.TemporaryDirectory(prefix="steuermann-import-") as staging:
            staging_path = Path(staging)
            tf.extractall(staging_path, filter="data")
            extracted = staging_path / profile_root
            if not extracted.exists() or not extracted.is_dir():
                print("Imported bundle missing profile root directory", file=sys.stderr)
                return 1
            shutil.move(str(extracted), str(target))

    missing = _validate_profile_files(target)
    if missing:
        print(f"Imported profile missing files: {', '.join(missing)}", file=sys.stderr)
        return 1

    profile_yaml = target / "profile.yaml"
    profile_data = yaml.safe_load(profile_yaml.read_text(encoding="utf-8")) or {}

    # Keep metadata consistent with directory name so profile validation passes
    # after importing into a differently named target folder.
    profile_data["profile_id"] = args.profile
    profile_yaml.write_text(
        yaml.safe_dump(profile_data, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )

    missing_required_keys = [key for key in required_keys if not _dig_optional(profile_data, key)[0]]
    if missing_required_keys:
        print(
            f"Imported profile missing required metadata keys: {', '.join(missing_required_keys)}",
            file=sys.stderr,
        )
        return 1

    _print_payload(
        {
            "status": "ok",
            "target": str(target),
            "compatibility": {
                "framework_version": current_version,
                "framework_version_range": framework_version_range,
                "minimum_required_keys": required_keys,
            },
        },
        args.format,
    )
    return 0


def cmd_docs_check(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    for doc in DOCS_CONFORMANCE_FILES:
        exists = doc.exists()
        details = "Found" if exists else "Missing"
        checks.append({"path": str(doc), "status": "ok" if exists else "fail", "details": details})

    # Ensure canonical command is present in docs index.
    index_path = Path("docs/index.md")
    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        has_reference = "steuermann" in content
        checks.append(
            {
                "path": str(index_path),
                "status": "ok" if has_reference else "fail",
                "details": "Contains steuermann references" if has_reference else "Missing steuermann command references",
            }
        )

    configuration_path = Path("docs/configuration.md")
    if configuration_path.exists():
        configuration_text = configuration_path.read_text(encoding="utf-8")
        has_precedence = "repository defaults" in configuration_text and "profile overlay" in configuration_text and "environment variables" in configuration_text
        checks.append(
            {
                "path": str(configuration_path),
                "status": "ok" if has_precedence else "fail",
                "details": "Contains precedence model guidance" if has_precedence else "Missing precedence model guidance",
            }
        )

    contract_checks = _contract_check_payload()["checks"]
    checks.extend(contract_checks)

    has_fail = any(c["status"] == "fail" for c in checks)

    def classify_domain(item: dict[str, Any]) -> str:
        path = str(item.get("path", ""))
        details = str(item.get("details", ""))
        if path.startswith("docs/") or path == "README.md":
            return "docs"
        if path == str(CONTRACT_PATH) and details.startswith("profile_bundle_compatibility"):
            return "bundle-compat"
        if path == str(CONTRACT_PATH):
            return "contract"
        return "other"

    drift_items = []
    by_domain: dict[str, int] = {"docs": 0, "contract": 0, "bundle-compat": 0, "other": 0}
    for item in checks:
        if item.get("status") != "fail":
            continue
        domain = classify_domain(item)
        by_domain[domain] = by_domain.get(domain, 0) + 1
        drift_items.append(
            {
                "path": str(item.get("path", "")),
                "status": item.get("status", "fail"),
                "details": item.get("details", ""),
                "severity": "error" if args.strict else "warning",
                "domain": domain,
            }
        )
    payload = {
        "status": "error" if has_fail else "ok",
        "checks": checks,
        "drift_report": {
            "total": len(drift_items),
            "items": drift_items,
            "docs_checks": len([i for i in checks if str(i.get("path", "")).startswith("docs/") or i.get("path") == "README.md"]),
            "contract_checks": len(contract_checks),
            "by_domain": by_domain,
        },
    }
    _print_payload(payload, args.format)
    return 1 if (has_fail and args.strict) else 0


def _add_common_format_arg(parser: argparse.ArgumentParser, default: str = "yaml") -> None:
    parser.add_argument("--format", choices=["yaml", "json"], default=default, help="Output format")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Steuermann operations CLI")
    subparsers = parser.add_subparsers(dest="command")

    profile_parser = subparsers.add_parser("profile", help="Profile operations")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command")

    active_parser = profile_subparsers.add_parser("active", help="Show active profile")
    active_parser.add_argument("--profile", help="Override PROFILE_ID for this command")
    _add_common_format_arg(active_parser)
    active_parser.set_defaults(func=cmd_profile_active)

    scaffold_parser = profile_subparsers.add_parser("scaffold", help="Create profile scaffold")
    scaffold_parser.add_argument("--from", dest="from_profile", required=True, help="Source profile template")
    scaffold_parser.add_argument("--profile", required=True, help="Target profile id (directory under config/profiles)")
    _add_common_format_arg(scaffold_parser)
    scaffold_parser.set_defaults(func=cmd_profile_scaffold)

    bundle_parser = profile_subparsers.add_parser("bundle", help="Bundle profile import/export")
    bundle_subparsers = bundle_parser.add_subparsers(dest="bundle_command")

    export_parser = bundle_subparsers.add_parser("export", help="Export profile bundle")
    export_parser.add_argument("--profile", required=True, help="Profile id to export")
    export_parser.add_argument("--out", required=True, help="Bundle output path (.tar.gz)")
    _add_common_format_arg(export_parser)
    export_parser.set_defaults(func=cmd_profile_bundle_export)

    import_parser = bundle_subparsers.add_parser("import", help="Import profile bundle")
    import_parser.add_argument("--bundle", required=True, help="Bundle path (.tar.gz)")
    import_parser.add_argument("--profile", required=True, help="Target profile id (directory under config/profiles)")
    _add_common_format_arg(import_parser)
    import_parser.set_defaults(func=cmd_profile_bundle_import)

    config_parser = subparsers.add_parser("config", help="Config operations")
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    show_parser = config_subparsers.add_parser("show", help="Show effective config")
    show_parser.add_argument("--profile", help="Override PROFILE_ID for this command")
    show_parser.add_argument(
        "--section",
        choices=["core", "features", "tools", "agents", "ui", "profile_metadata"],
        help="Optional section filter",
    )
    _add_common_format_arg(show_parser)
    show_parser.set_defaults(func=cmd_config_show)

    explain_parser = config_subparsers.add_parser("explain", help="Explain effective key value")
    explain_parser.add_argument("--profile", help="Override PROFILE_ID for this command")
    explain_parser.add_argument("--key", required=True, help="Dot path key (for example core.rag.collection_name)")
    _add_common_format_arg(explain_parser)
    explain_parser.set_defaults(func=cmd_config_explain)

    validate_parser = config_subparsers.add_parser("validate", help="Validate config for profile(s)")
    validate_parser.add_argument("--profile", help="Profile to validate (default: base + all profiles)")
    validate_parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    _add_common_format_arg(validate_parser)
    validate_parser.set_defaults(func=cmd_config_validate)

    set_parser = config_subparsers.add_parser(
        "set",
        help="Set a profile-safe key in profile core overlay (dry-run by default)",
    )
    set_parser.add_argument("--profile", required=True, help="Target profile id (base is not allowed)")
    set_parser.add_argument("--key", required=True, help="Dot path key, for example core.llm.temperature")
    set_parser.add_argument("--value", required=True, help="New value parsed as YAML scalar/object")
    set_parser.add_argument("--apply", action="store_true", help="Persist the change (default: dry-run)")
    set_parser.add_argument("--confirm", default="", help="Required for --apply: must be APPLY")
    _add_common_format_arg(set_parser)
    set_parser.set_defaults(func=cmd_config_set)

    unset_parser = config_subparsers.add_parser(
        "unset",
        help="Unset a profile-safe key in profile core overlay (dry-run by default)",
    )
    unset_parser.add_argument("--profile", required=True, help="Target profile id (base is not allowed)")
    unset_parser.add_argument("--key", required=True, help="Dot path key, for example core.llm.temperature")
    unset_parser.add_argument("--apply", action="store_true", help="Persist the change (default: dry-run)")
    unset_parser.add_argument("--confirm", default="", help="Required for --apply: must be APPLY")
    _add_common_format_arg(unset_parser)
    unset_parser.set_defaults(func=cmd_config_unset)

    contract_parser = config_subparsers.add_parser("contract-check", help="Validate CLI contract parity")
    _add_common_format_arg(contract_parser)
    contract_parser.set_defaults(func=cmd_config_contract_check)

    setup_parser = subparsers.add_parser("setup", help="Setup diagnostics")
    setup_subparsers = setup_parser.add_subparsers(dest="setup_command")

    doctor_parser = setup_subparsers.add_parser("doctor", help="Run setup checks")
    doctor_parser.add_argument("--probe-endpoints", action="store_true", help="Probe configured endpoints")
    _add_common_format_arg(doctor_parser)
    doctor_parser.set_defaults(func=cmd_setup_doctor)

    docs_parser = subparsers.add_parser("docs", help="Docs conformance checks")
    docs_subparsers = docs_parser.add_subparsers(dest="docs_command")

    check_parser = docs_subparsers.add_parser("check", help="Check documentation conformance")
    check_parser.add_argument("--strict", action="store_true", help="Fail on detected drift")
    _add_common_format_arg(check_parser)
    check_parser.set_defaults(func=cmd_docs_check)

    ingest_parser = subparsers.add_parser("ingest", help="Ingestion commands")
    ingest_subparsers = ingest_parser.add_subparsers(dest="ingest_command")
    add_ingest_subcommands(ingest_subparsers)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return int(args.func(args))
    except Exception as exc:
        if hasattr(args, "format") and getattr(args, "format") == "json":
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

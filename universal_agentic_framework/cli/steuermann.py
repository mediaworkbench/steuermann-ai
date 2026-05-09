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
from typing import Any, Iterable

import yaml

from universal_agentic_framework.cli.ingest import add_ingest_subcommands
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


def _with_profile_env(profile_id: str | None) -> dict[str, str]:
    env = dict(os.environ)
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


def cmd_config_contract_check(args: argparse.Namespace) -> int:
    payload = _contract_check_payload()
    _print_payload(payload, args.format)
    return 1 if payload["status"] == "error" else 0


def cmd_setup_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, blocking: bool, details: str) -> None:
        checks.append({
            "name": name,
            "status": "ok" if ok else "fail",
            "blocking": blocking,
            "details": details,
        })

    postgres_password = os.getenv("POSTGRES_PASSWORD")
    add_check(
        "POSTGRES_PASSWORD",
        bool(postgres_password),
        True,
        "Set POSTGRES_PASSWORD in .env or environment" if not postgres_password else "Present",
    )

    profile_id = get_active_profile_id()
    profile_dir = get_profile_dir(profile_id=profile_id, require_exists=False)
    profile_ok = profile_id == "base" or (profile_dir is not None and profile_dir.exists())
    add_check(
        "PROFILE_ID",
        profile_ok,
        True,
        "Profile directory missing" if not profile_ok else f"Resolved profile: {profile_id}",
    )

    ingest_collection = os.getenv("INGEST_COLLECTION")
    try:
        core = load_core_config()
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

        endpoint = os.getenv("LLM_ENDPOINT", "")
        ok = bool(endpoint)
        details = "LLM_ENDPOINT missing"
        if endpoint:
            try:
                with httpx.Client(timeout=3.0) as client:
                    response = client.get(endpoint)
                ok = response.status_code < 500
                details = f"HTTP {response.status_code}"
            except Exception as exc:
                ok = False
                details = str(exc)
        add_check("LLM endpoint probe", ok, False, details)

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

    target = Path(args.to)
    _copytree(starter_dir, target)

    profile_id = args.profile_id or target.name
    profile_yaml = target / "profile.yaml"
    if profile_yaml.exists():
        data = yaml.safe_load(profile_yaml.read_text(encoding="utf-8")) or {}
        data["profile_id"] = profile_id
        profile_yaml.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "profile_id": profile_id,
        "framework_version": "0.x",
        "required_files": PROFILE_REQUIRED_FILES + ["prompts/"],
    }
    (target / "bundle_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )

    _print_payload({"status": "ok", "target": str(target), "profile_id": profile_id}, args.format)
    return 0


def _bundle_manifest(profile_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "framework_version_range": ">=0.2,<1.0",
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
            _bundle_manifest(args.profile), sort_keys=True, allow_unicode=False
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

    target = Path(args.to)
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

        members = [n for n in names if n != "bundle_manifest.yaml"]
        root_dirs = sorted({n.split("/")[0] for n in members if "/" in n})
        if not root_dirs:
            print("Invalid bundle content", file=sys.stderr)
            return 1

        profile_root = root_dirs[0]
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        tf.extractall(parent)

    extracted = parent / profile_root
    extracted.rename(target)

    missing = _validate_profile_files(target)
    if missing:
        print(f"Imported profile missing files: {', '.join(missing)}", file=sys.stderr)
        return 1

    _print_payload({"status": "ok", "target": str(target)}, args.format)
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

    checks.extend(_contract_check_payload()["checks"])

    has_fail = any(c["status"] == "fail" for c in checks)
    payload = {"status": "error" if has_fail else "ok", "checks": checks}
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
    scaffold_parser.add_argument("--to", required=True, help="Target directory")
    scaffold_parser.add_argument("--profile-id", help="Override profile_id in generated profile.yaml")
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
    import_parser.add_argument("--to", required=True, help="Target directory")
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

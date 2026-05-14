"""Config loaders with hierarchical merge and env substitution."""
from __future__ import annotations

import os
import re
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Type, TypeVar

import yaml

from .schemas import (
    AgentsConfig,
    CoreConfig,
    FeaturesConfig,
    ProfileMetadata,
    ProfileUIConfig,
    ToolsConfig,
    PROFILE_ID_PATTERN,
)

T = TypeVar("T", CoreConfig, AgentsConfig, ToolsConfig, FeaturesConfig)
_PROFILE_ID_RE = re.compile(PROFILE_ID_PATTERN)
_DEFAULT_PROFILES_DIR = Path("config/profiles")
_LEGACY_PROFILES_DIR = Path("profiles")
_DISALLOWED_PROFILE_FEATURE_FLAGS = {"ingestion_service", "authentication", "monitoring"}
_PROFILE_CORE_ALLOWED_PREFIXES = {
    "fork.language",
    "fork.locale",
    "fork.timezone",
    "fork.supported_languages",
    "llm",
    "ingestion",
    "prompts",
    "tool_routing",
    "rag",
    "tokens",
    "memory.embeddings",
    "memory.mem0",
    "memory.retention",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_active_profile_id(env: Optional[Mapping[str, str]] = None) -> str:
    env_map = env or os.environ
    profile_id = str(env_map.get("PROFILE_ID", "") or "").strip()
    if not profile_id:
        raise ValueError("PROFILE_ID must be set to an active profile id")
    if not _PROFILE_ID_RE.match(profile_id):
        raise ValueError(f"Invalid PROFILE_ID '{profile_id}'")
    if profile_id == "base":
        raise ValueError("PROFILE_ID='base' is no longer a valid runtime profile")
    return profile_id


def get_profile_dir(
    profiles_dir: Path | str = _DEFAULT_PROFILES_DIR,
    profile_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    *,
    require_exists: bool = False,
) -> Optional[Path]:
    resolved_profile_id = profile_id or get_active_profile_id(env)
    profiles_root = Path(profiles_dir)
    if profiles_root == _DEFAULT_PROFILES_DIR and not profiles_root.exists() and _LEGACY_PROFILES_DIR.exists():
        profiles_root = _LEGACY_PROFILES_DIR
    path = profiles_root / resolved_profile_id
    if require_exists and not path.exists():
        raise FileNotFoundError(f"Active profile '{resolved_profile_id}' was selected but {path} does not exist")
    return path


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _merge_named_list(
    base_items: Iterable[Dict[str, Any]],
    override_items: Iterable[Dict[str, Any]],
    *,
    key: str,
) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    index_by_key: dict[str, int] = {}

    for item in base_items:
        item_key = item.get(key)
        if not item_key:
            continue
        index_by_key[str(item_key)] = len(merged)
        merged.append(dict(item))

    for item in override_items:
        item_key = item.get(key)
        if not item_key:
            continue
        normalized_key = str(item_key)
        if normalized_key in index_by_key:
            existing = merged[index_by_key[normalized_key]]
            _deep_merge(existing, item)
        else:
            index_by_key[normalized_key] = len(merged)
            merged.append(dict(item))

    return merged


def _flatten_leaf_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_flatten_leaf_paths(child, child_prefix))
        return paths
    return [prefix]


def _validate_profile_core_overlay(overlay_data: Mapping[str, Any], profile_id: str) -> None:
    for path in _flatten_leaf_paths(overlay_data):
        if any(path == allowed or path.startswith(f"{allowed}.") for allowed in _PROFILE_CORE_ALLOWED_PREFIXES):
            continue
        raise ValueError(
            f"Profile '{profile_id}' cannot override deployment-global core setting '{path}'"
        )


def _validate_profile_features_overlay(overlay_data: Mapping[str, Any], profile_id: str) -> None:
    disallowed = sorted(key for key in overlay_data.keys() if key in _DISALLOWED_PROFILE_FEATURE_FLAGS)
    if disallowed:
        joined = ", ".join(disallowed)
        raise ValueError(f"Profile '{profile_id}' cannot override deployment-global feature flags: {joined}")


def _merge_config_data(filename: str, base_data: Dict[str, Any], override_data: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base_data)

    if filename == "tools.yaml":
        _deep_merge(merged, {k: v for k, v in override_data.items() if k != "tools"})
        merged["tools"] = _merge_named_list(base_data.get("tools", []), override_data.get("tools", []), key="name")
        return merged

    if filename == "agents.yaml":
        _deep_merge(merged, {k: v for k, v in override_data.items() if k != "crew_chains"})
        merged["crew_chains"] = _merge_named_list(
            base_data.get("crew_chains", []),
            override_data.get("crew_chains", []),
            key="name",
        )
        return merged

    _deep_merge(merged, override_data)
    return merged


def _substitute_env(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return Template(value).safe_substitute(env)
    if isinstance(value, list):
        return [_substitute_env(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _substitute_env(v, env) for k, v in value.items()}
    return value


def _load_prompt_files(prompts_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load per-language prompt YAML files from a prompts/ directory.

    Returns ``{lang_code: {prompt_type: text, ...}, ...}``.
    """
    result: Dict[str, Dict[str, str]] = {}
    if not prompts_dir.is_dir():
        return result
    for yaml_path in sorted(prompts_dir.glob("*.yaml")):
        lang_code = yaml_path.stem
        data = _load_yaml(yaml_path)
        if data and isinstance(data, dict):
            result[lang_code] = {k: str(v) for k, v in data.items() if isinstance(v, str)}
    return result


def _merge_prompt_languages(
    base: Dict[str, Dict[str, str]],
    overlay: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    """Merge two prompt-language dicts (overlay wins per-key)."""
    merged = {lang: dict(prompts) for lang, prompts in base.items()}
    for lang, prompts in overlay.items():
        if lang in merged:
            merged[lang].update(prompts)
        else:
            merged[lang] = dict(prompts)
    return merged


def _load_config(
    target: Type[T],
    filename: str,
    config_dir: Path,
    base_dir: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    profiles_dir: Path | str = _DEFAULT_PROFILES_DIR,
) -> T:
    env_map: Mapping[str, str] = env or os.environ
    profile_id = get_active_profile_id(env_map)

    data: Dict[str, Any] = {}
    if base_dir:
        base_data = _load_yaml(base_dir / filename)
        _deep_merge(data, base_data)

    override_data = _load_yaml(config_dir / filename)
    data = _merge_config_data(filename, data, override_data)

    profile_dir = get_profile_dir(profiles_dir, profile_id=profile_id, env=env_map, require_exists=True)
    assert profile_dir is not None
    profile_data = _load_yaml(profile_dir / filename)
    if filename == "core.yaml":
        _validate_profile_core_overlay(profile_data, profile_id)
    elif filename == "features.yaml":
        _validate_profile_features_overlay(profile_data, profile_id)
    data = _merge_config_data(filename, data, profile_data)

    # Load per-language prompt files for core config
    if filename == "core.yaml":
        base_prompts = _load_prompt_files(config_dir / "prompts")
        profile_prompts = _load_prompt_files(profile_dir / "prompts")
        merged_prompts = _merge_prompt_languages(base_prompts, profile_prompts)
        if merged_prompts:
            prompts_section = data.setdefault("prompts", {})
            prompts_section["languages"] = merged_prompts

    substituted = _substitute_env(data, env_map)
    return target.model_validate(substituted)


def load_profile_metadata(
    profiles_dir: Path | str = _DEFAULT_PROFILES_DIR,
    profile_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[ProfileMetadata]:
    env_map = env or os.environ
    resolved_profile_id = profile_id or get_active_profile_id(env_map)

    profile_dir = get_profile_dir(profiles_dir, profile_id=resolved_profile_id, env=env_map, require_exists=True)
    assert profile_dir is not None
    metadata_path = profile_dir / "profile.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Active profile '{resolved_profile_id}' requires {metadata_path}"
        )

    metadata = ProfileMetadata.model_validate(_load_yaml(metadata_path))
    if metadata.profile_id != resolved_profile_id:
        raise ValueError(
            f"Profile metadata id '{metadata.profile_id}' does not match directory '{resolved_profile_id}'"
        )
    return metadata


def load_profile_ui_config(
    profiles_dir: Path | str = _DEFAULT_PROFILES_DIR,
    profile_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> ProfileUIConfig:
    env_map = env or os.environ
    resolved_profile_id = profile_id or get_active_profile_id(env_map)
    profile_dir = get_profile_dir(profiles_dir, profile_id=resolved_profile_id, env=env_map, require_exists=True)
    assert profile_dir is not None
    return ProfileUIConfig.model_validate(_load_yaml(profile_dir / "ui.yaml"))


def load_core_config(
    config_dir: Path | str = Path("config"),
    base_dir: Optional[Path | str] = None,
    env: Optional[Mapping[str, str]] = None,
    profiles_dir: Path | str = _DEFAULT_PROFILES_DIR,
) -> CoreConfig:
    return _load_config(CoreConfig, "core.yaml", Path(config_dir), Path(base_dir) if base_dir else None, env, profiles_dir)


def load_agents_config(
    config_dir: Path | str = Path("config"),
    base_dir: Optional[Path | str] = None,
    env: Optional[Mapping[str, str]] = None,
    profiles_dir: Path | str = _DEFAULT_PROFILES_DIR,
) -> AgentsConfig:
    return _load_config(AgentsConfig, "agents.yaml", Path(config_dir), Path(base_dir) if base_dir else None, env, profiles_dir)


def load_tools_config(
    config_dir: Path | str = Path("config"),
    base_dir: Optional[Path | str] = None,
    env: Optional[Mapping[str, str]] = None,
    profiles_dir: Path | str = _DEFAULT_PROFILES_DIR,
) -> ToolsConfig:
    return _load_config(ToolsConfig, "tools.yaml", Path(config_dir), Path(base_dir) if base_dir else None, env, profiles_dir)


def load_features_config(
    config_dir: Path | str = Path("config"),
    base_dir: Optional[Path | str] = None,
    env: Optional[Mapping[str, str]] = None,
    profiles_dir: Path | str = _DEFAULT_PROFILES_DIR,
) -> FeaturesConfig:
    return _load_config(FeaturesConfig, "features.yaml", Path(config_dir), Path(base_dir) if base_dir else None, env, profiles_dir)

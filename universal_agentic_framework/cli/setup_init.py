"""First-time setup wizard for steuermann (``steuermann setup init``).

A single interactive command that replaces the manual onboarding sequence:
choose and fully configure an LLM provider, generate strong secrets + an argon2
admin password hash, write a valid ``.env`` (preserving the ``.env.example``
comments), create/chown the data directories, run the existing pre-flight
validation, and print a one-time summary with the generated credentials.

This module is a sibling of ``ingest.py`` and is registered by ``steuermann.py``
via :func:`add_setup_init_subcommand`. To avoid an import cycle (``steuermann``
imports this module at top level), every ``steuermann`` helper is imported
**lazily inside the functions that use it**.
"""

from __future__ import annotations

import argparse
import copy
import getpass
import json
import os
from pathlib import Path
import platform
import re
import secrets
import sys
import time
from typing import Any
import urllib.error
import urllib.request

import yaml

# --- Defaults (mirror config/profiles/starter/core.yaml + .env.example) -------

DEFAULT_PROFILE = "starter"
DEFAULT_SCAFFOLD_TARGET = "local"
DEFAULT_CHAT_MODEL = "openai/google/gemma-4-e4b"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-granite-embedding-278m-multilingual"
DEFAULT_EMBEDDING_ENDPOINT = "http://host.docker.internal:1234/v1"
DEFAULT_EMBEDDING_DIMENSION = 768

PROFILE_ID_RE = re.compile(r"^[a-z0-9_-]+$")

# Provider → env var names + defaults (source: .env.example lines 40-44).
PROVIDER_ENV: dict[str, dict[str, str]] = {
    "lmstudio": {
        "api_base_var": "LLM_PROVIDERS_LMSTUDIO_API_BASE",
        "default_api_base": "http://host.docker.internal:1234/v1",
        "api_key_var": "OPENAI_API_KEY",
        "default_api_key": "lm-studio",
    },
    "ollama": {
        "api_base_var": "LLM_PROVIDERS_OLLAMA_API_BASE",
        "default_api_base": "http://host.docker.internal:11434/v1",
        "api_key_var": "OPENAI_API_KEY",
        "default_api_key": "lm-studio",
    },
    "openrouter": {
        "api_base_var": "LLM_PROVIDERS_OPENROUTER_API_BASE",
        "default_api_base": "https://openrouter.ai/api/v1",
        "api_key_var": "LLM_PROVIDERS_OPENROUTER_API_KEY",
        "default_api_key": "",
    },
}
PROVIDER_CHOICES = list(PROVIDER_ENV.keys())

# Providers whose endpoint exposes an OpenAI-compatible ``GET /v1/models`` we can
# enumerate to offer an interactive pick-list. (OpenRouter also exposes one, but
# its catalog is hundreds of entries — not useful as a scrolling terminal list.)
LISTABLE_PROVIDERS = frozenset({"lmstudio", "ollama"})

# LiteLLM transport prefixes we must not double-up. A model id lacking any of
# these gets ``openai/`` prepended (see :func:`ensure_openai_prefix`).
_KNOWN_MODEL_PREFIXES = ("openai/", "ollama/", "lm_studio/", "lmstudio/", "openrouter/")


# --- Narration ---------------------------------------------------------------
# Human-facing narration goes to stderr so stdout carries only the final
# structured summary (machine-parseable for --format json/yaml).


def _say(message: str = "") -> None:
    print(message, file=sys.stderr)


# --- Platform / secrets ------------------------------------------------------


def detect_platform() -> dict[str, Any]:
    """Describe the host OS/arch and current uid/gid (None on Windows)."""
    system = platform.system()
    is_linux = system == "Linux"
    machine = platform.machine().lower()
    is_arm = machine in {"aarch64", "arm64"}

    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    uid = getuid() if callable(getuid) else None
    gid = getgid() if callable(getgid) else None

    return {
        "os": system,
        "machine": machine,
        "is_linux": is_linux,
        "is_arm": is_arm,
        "uid": uid,
        "gid": gid,
        # haktansuren/qdrant-pi5-fixed-jemalloc note (docs/configuration.md).
        "pi5_qdrant_warning": is_arm and is_linux,
    }


def generate_postgres_password() -> str:
    """Strong, URL-safe Postgres password (never the insecure 'framework')."""
    return secrets.token_urlsafe(24)


def generate_token_hex() -> str:
    """64-char hex token, used for AUTH_SESSION_SECRET and CHAT_ACCESS_TOKEN."""
    return secrets.token_hex(32)


def generate_password_hash(plaintext: str) -> str | None:
    """argon2id hash of ``plaintext``; ``None`` when argon2-cffi is unavailable."""
    try:
        from backend.auth import hash_password

        return hash_password(plaintext)
    except (ImportError, RuntimeError):
        return None


# --- TTY-degrading prompt helpers --------------------------------------------


def _is_tty() -> bool:
    return sys.stdin.isatty()


def _prompt_with_default(label: str, default: str) -> str:
    if not _is_tty():
        return default
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw else default


def _prompt_bool(label: str, default: bool) -> bool:
    if not _is_tty():
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{label} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "true", "1"}


def _prompt_choice(label: str, choices: list[str], default: str) -> str:
    if not _is_tty():
        return default
    raw = input(f"{label} {choices} [{default}]: ").strip()
    if not raw:
        return default
    return raw if raw in choices else default


def _prompt_password(label: str) -> str | None:
    """Return the entered password, or ``None`` to signal auto-generation."""
    if not _is_tty():
        return None
    return getpass.getpass(f"{label}: ") or None


# --- .env helpers ------------------------------------------------------------


def _parse_env_value(raw: str) -> str:
    """Parse a single .env value, stripping quotes and trailing inline comments.

    Mirrors ``steuermann._parse_env_value`` — duplicated here to avoid an
    import cycle (``steuermann`` imports this module at top level).
    """
    v = raw.strip()
    if v.startswith('"'):
        end = v.find('"', 1)
        return v[1:end] if end != -1 else v[1:]
    if v.startswith("'"):
        end = v.find("'", 1)
        return v[1:end] if end != -1 else v[1:]
    for sep in (" #", "\t#"):
        idx = v.find(sep)
        if idx != -1:
            v = v[:idx]
    return v.strip()


def _read_env_file_values(path: Path) -> dict[str, str]:
    """Read only the values written in ``path`` (no process environment).

    Mirrors the parse rule of ``steuermann._load_env_file`` but does not merge
    ``os.environ`` — used for value preservation (e.g. POSTGRES_PASSWORD) so a
    re-run keeps what the file actually holds.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _parse_env_value(value)
    return values


def render_env_from_example(
    example_text: str,
    overrides: dict[str, str],
    *,
    quoted_keys: frozenset[str] = frozenset({"AUTH_PASSWORD_HASH"}),
) -> tuple[str, list[str]]:
    """Render a ``.env`` from ``.env.example`` text, replacing only known values.

    Comments and blank lines are preserved verbatim. For a ``KEY=...`` line whose
    ``KEY`` is in ``overrides`` only the value is replaced. Keys in ``quoted_keys``
    are wrapped in single quotes (literal ``$`` for Docker Compose). Override keys
    not present in the example are appended in a trailing block.

    Returns ``(rendered_text, unmatched_override_keys)``.
    """

    def render_line(key: str, value: str) -> str:
        if key in quoted_keys:
            return f"{key}='{value}'"
        return f"{key}={value}"

    used: set[str] = set()
    out_lines: list[str] = []
    for raw_line in example_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(raw_line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in overrides:
            out_lines.append(render_line(key, overrides[key]))
            used.add(key)
        else:
            out_lines.append(raw_line)

    extras = [key for key in overrides if key not in used]
    if extras:
        out_lines.append("")
        out_lines.append("# Added by steuermann setup init")
        for key in extras:
            out_lines.append(render_line(key, overrides[key]))

    return "\n".join(out_lines) + "\n", extras


def write_env_file(path: Path, content: str) -> Path | None:
    """Write ``content`` to ``path``, backing up any existing file first.

    An existing ``.env`` is always copied to ``.env.bak`` (timestamped
    ``.env.bak.<ts>`` if ``.env.bak`` already exists) so the operator's existing
    values/secrets are never silently clobbered. Returns the backup path, if any.
    """
    backup_path: Path | None = None
    if path.exists():
        candidate = path.parent / (path.name + ".bak")
        if candidate.exists():
            ts = time.strftime("%Y%m%d%H%M%S")
            candidate = path.parent / (path.name + f".bak.{ts}")
        candidate.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        backup_path = candidate
    path.write_text(content, encoding="utf-8")
    return backup_path


# --- Data directories --------------------------------------------------------


def ensure_data_dirs(uid: int | None, gid: int | None, *, is_linux: bool) -> dict[str, Any]:
    """Create the docker-compose bind-mount dirs and (on Linux) chown them."""
    dirs = [Path("./data/rag-data"), Path("./data/workspaces"), Path("./data/checkpoints")]
    created: list[str] = []
    chowned: list[str] = []
    status = "ok"
    note: str | None = None

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        created.append(str(directory))
        if is_linux and uid is not None and gid is not None:
            try:
                os.chown(directory, uid, gid)
                chowned.append(str(directory))
            except OSError as exc:
                status = "chown-failed"
                note = (
                    f"Could not chown {directory}: {exc}. "
                    f"Run: sudo chown -R {uid}:{gid} ./data"
                )

    return {"status": status, "dirs": created, "chowned": chowned, "note": note}


# --- Profile resolution + scaffolding ----------------------------------------


def resolve_profile(args: argparse.Namespace) -> tuple[str, list[str], bool, str | None]:
    """Return ``(profile_id, available, exists, error)`` for the requested profile.

    ``error`` is set only for an invalid id (bad regex / ``base``). A valid but
    non-existent id is allowed — it will be scaffolded from starter.
    """
    from universal_agentic_framework.cli import steuermann

    available = [name for name in steuermann._iter_profiles() if name != "base"]
    profile_id = (getattr(args, "profile", None) or DEFAULT_PROFILE).strip()

    if profile_id == "base" or not PROFILE_ID_RE.match(profile_id):
        return profile_id, available, False, f"Invalid profile id: {profile_id!r}"

    exists = (steuermann._profiles_root() / profile_id).is_dir()
    return profile_id, available, exists, None


def _scaffold_profile(from_profile: str, target: str) -> Path:
    """Copy ``from_profile`` → ``target`` and write a fresh bundle manifest.

    Mirrors ``cmd_profile_scaffold`` (reusing its ``_copytree`` + manifest path)
    without printing, so the wizard's single structured summary stays intact.
    """
    from universal_agentic_framework.cli import steuermann

    src = steuermann._profiles_root() / from_profile
    dst = steuermann._profiles_root() / target
    steuermann._copytree(src, dst)  # raises FileExistsError if dst exists

    profile_yaml = dst / "profile.yaml"
    if profile_yaml.exists():
        data = yaml.safe_load(profile_yaml.read_text(encoding="utf-8")) or {}
        data["profile_id"] = target
        profile_yaml.write_text(
            yaml.safe_dump(data, sort_keys=True, allow_unicode=False), encoding="utf-8"
        )

    manifest = {
        "schema_version": 1,
        "manifest_version": 1,
        "profile_id": target,
        "compatibility": steuermann._profile_compatibility(dst, target),
        "required_files": steuermann.PROFILE_REQUIRED_FILES + ["prompts/"],
    }
    (dst / "bundle_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=True, allow_unicode=False), encoding="utf-8"
    )
    return dst


def _resolve_scaffold_target(args: argparse.Namespace) -> str | None:
    """Pick a fresh, non-colliding profile id for a customized starter copy.

    Interactive: prompt (default ``local``), re-prompting on collision/invalid id.
    Non-interactive: return ``local`` if free, else ``None`` to signal abort.
    """
    from universal_agentic_framework.cli import steuermann

    root = steuermann._profiles_root()

    if not _is_tty():
        return None if (root / DEFAULT_SCAFFOLD_TARGET).exists() else DEFAULT_SCAFFOLD_TARGET

    while True:
        candidate = _prompt_with_default(
            "New profile id for your customized config", DEFAULT_SCAFFOLD_TARGET
        ).strip()
        if candidate in ("", "base") or not PROFILE_ID_RE.match(candidate):
            _say("  ⚠️  Invalid id; use lowercase letters, digits, '-' or '_'.")
            continue
        if (root / candidate).exists():
            _say(f"  ⚠️  Profile '{candidate}' already exists; choose another id.")
            continue
        return candidate


def maybe_scaffold_for_customization(
    target: str, source_exists: bool, changed: bool, *, args: argparse.Namespace
) -> tuple[str | None, str | None, str | None]:
    """Decide the final profile id, scaffolding from starter when needed.

    Returns ``(final_profile_id, scaffolded_from, error)``:

    * ``target == "starter"`` and unchanged → use starter as-is.
    * ``target == "starter"`` and changed → scaffold a fresh copy (default
      ``local``) so starter stays pristine.
    * a new (non-existent) name → scaffold from starter.
    * an existing non-starter profile → edit in place.

    On a scaffold-id collision in non-interactive mode, ``error`` is set so the
    caller can abort before writing anything.
    """
    if target == DEFAULT_PROFILE:
        if not changed:
            return DEFAULT_PROFILE, None, None
        new_id = _resolve_scaffold_target(args)
        if new_id is None:
            return (
                None,
                None,
                f"Scaffold target '{DEFAULT_SCAFFOLD_TARGET}' already exists; "
                "pass --profile <new-id> (non-interactive)",
            )
        try:
            _scaffold_profile(DEFAULT_PROFILE, new_id)
        except FileExistsError:
            return None, None, f"Profile '{new_id}' already exists"
        return new_id, DEFAULT_PROFILE, None

    if not source_exists:
        try:
            _scaffold_profile(DEFAULT_PROFILE, target)
        except FileExistsError:
            return None, None, f"Profile '{target}' already exists"
        return target, DEFAULT_PROFILE, None

    return target, None, None


# --- Provider / model / embedding collection ---------------------------------


def provider_env_overrides(provider: str, api_base: str, api_key: str) -> dict[str, str]:
    """Map a provider's endpoint + key to the matching ``.env`` variables."""
    penv = PROVIDER_ENV[provider]
    return {
        penv["api_base_var"]: api_base,
        penv["api_key_var"]: api_key if api_key else penv["default_api_key"],
    }


def ensure_openai_prefix(model: str) -> str:
    """Prepend the LiteLLM ``openai/`` transport prefix unless one is already present.

    LM Studio and Ollama are both reached over their OpenAI-compatible ``/v1``
    endpoint, so their native ids (``gemma4:e2b``, ``google/gemma-4-e2b``) must be
    written to config as ``openai/<id>``. Without it LiteLLM would parse a bare
    ``google/…`` as the *google* provider (Gemini) rather than the local server.
    """
    stripped = model.strip()
    if not stripped:
        return stripped
    if any(stripped.startswith(prefix) for prefix in _KNOWN_MODEL_PREFIXES):
        return stripped
    return f"openai/{stripped}"


def fetch_provider_models(endpoint: str, api_key: str, *, timeout: float = 4.0) -> list[str]:
    """Return de-duplicated, sorted model ids from ``GET {endpoint}/models``.

    Works against any OpenAI-compatible server (LM Studio, Ollama's ``/v1``).
    ``host.docker.internal`` is rewritten to ``localhost`` because the wizard runs
    on the host, where that docker-only alias does not resolve. Returns ``[]`` on
    any network/parse error so callers fall back to a free-text prompt.
    """
    probe = endpoint.replace("host.docker.internal", "localhost").rstrip("/")
    request = urllib.request.Request(probe + "/models", headers={"Accept": "application/json"})
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids = [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]
    return sorted(dict.fromkeys(ids))


def _prompt_model_from_list(label: str, models: list[str], default: str) -> str:
    """Prompt for a model id, offering ``models`` as a numbered pick-list.

    ``default`` and the return value are BARE ids (no transport prefix). Accepts a
    list number, a typed id (custom), or Enter (keep ``default``). Falls back to a
    plain default prompt when non-interactive or the list is empty.
    """
    if not _is_tty() or not models:
        return _prompt_with_default(label, default)

    _say(f"{label} — available on this server:")
    for index, name in enumerate(models, 1):
        marker = "  ← current" if name == default else ""
        _say(f"   {index:2}. {name}{marker}")
    raw = input(f"{label} — number, id, or Enter for [{default}]: ").strip()
    if not raw:
        return default
    if raw.isdigit():
        choice = int(raw)
        if 1 <= choice <= len(models):
            return models[choice - 1]
        _say(f"   ⚠️  {choice} is out of range; keeping {default!r}.")
        return default
    return raw  # treated as a custom id


def prompt_provider_config(
    provider: str,
    current_core: dict[str, Any],
    current_env: dict[str, str],
    *,
    endpoint: str,
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Collect per-role models + embedding settings, pre-filled from the source.

    For locally listable providers (LM Studio / Ollama) the chat/vision/auxiliary/
    embedding prompts offer a live pick-list fetched from ``{endpoint}/models``;
    the selected bare id is stored with the ``openai/`` transport prefix. Returns
    ``{chat, vision, auxiliary}`` role configs (with ``$``-placeholder
    ``api_base``/``api_key`` references) plus an ``embedding`` block
    ``{endpoint, model, dimension}``.
    """
    penv = PROVIDER_ENV[provider]
    api_base_ref = f"${penv['api_base_var']}"
    api_key_ref = f"${penv['api_key_var']}"

    roles_cur = (current_core.get("llm") or {}).get("roles") or {}

    # Offer a live pick-list of the server's models. Interactive-only (there is
    # nothing to pick non-interactively) and best-effort — a down/unreachable
    # server just degrades to the free-text prompt.
    models: list[str] = []
    if provider in LISTABLE_PROVIDERS and _is_tty():
        models = fetch_provider_models(endpoint, api_key)
        if models:
            _say(f"🔎 Found {len(models)} model(s) on the {provider} endpoint ({endpoint}).")
        else:
            _say(
                f"   ⚠️  Could not list models from {endpoint} — is {provider} reachable "
                "from this host? You can still type a model id manually."
            )

    chat_default = _bare_model((roles_cur.get("chat") or {}).get("model") or DEFAULT_CHAT_MODEL)
    chat_bare = _prompt_model_from_list("Chat model", models, chat_default)
    chat_model = ensure_openai_prefix(chat_bare)
    vision_default = _bare_model((roles_cur.get("vision") or {}).get("model") or chat_bare)
    vision_model = ensure_openai_prefix(
        _prompt_model_from_list("Vision model (Enter = chat model)", models, vision_default)
    )
    aux_default = _bare_model((roles_cur.get("auxiliary") or {}).get("model") or chat_bare)
    aux_model = ensure_openai_prefix(
        _prompt_model_from_list("Auxiliary model (Enter = chat model)", models, aux_default)
    )

    emb_role_cur = roles_cur.get("embedding") or {}
    emb_endpoint_default = current_env.get("EMBEDDING_SERVER") or DEFAULT_EMBEDDING_ENDPOINT
    emb_endpoint = _prompt_with_default("Embedding endpoint", emb_endpoint_default)
    # Reuse the fetched list when the embedding endpoint matches; otherwise probe
    # the (possibly different) embedding endpoint on its own.
    if emb_endpoint.rstrip("/") == endpoint.rstrip("/"):
        emb_models = models
    elif provider in LISTABLE_PROVIDERS and _is_tty():
        emb_models = fetch_provider_models(emb_endpoint, api_key)
    else:
        emb_models = []
    emb_model_default = _bare_model(emb_role_cur.get("model") or DEFAULT_EMBEDDING_MODEL)
    emb_model = ensure_openai_prefix(
        _prompt_model_from_list("Embedding model", emb_models, emb_model_default)
    )
    dim_default = str(
        (current_core.get("memory") or {}).get("embeddings", {}).get("dimension")
        or DEFAULT_EMBEDDING_DIMENSION
    )
    dim_raw = _prompt_with_default("Embedding dimension", dim_default)
    try:
        emb_dimension = int(dim_raw)
    except ValueError:
        emb_dimension = int(dim_default)

    def role(model: str) -> dict[str, str]:
        return {
            "provider_id": provider,
            "api_base": api_base_ref,
            "api_key": api_key_ref,
            "model": model,
        }

    return {
        "chat": role(chat_model),
        "vision": role(vision_model),
        "auxiliary": role(aux_model),
        "embedding": {"endpoint": emb_endpoint, "model": emb_model, "dimension": emb_dimension},
    }


def _config_differs_from_current(
    current_core: dict[str, Any],
    role_cfg: dict[str, Any],
    provider: str,
    current_env: dict[str, str],
) -> bool:
    """Did the operator deviate from the offered defaults?

    True when any value that the wizard would persist (provider, per-role models,
    embedding model/dimension, Mem0 provider) differs from the source profile, or
    when the embedding endpoint differs from the currently configured one. The
    embedding endpoint matters because the wizard normalizes
    ``llm.roles.embedding.api_base`` to ``$EMBEDDING_SERVER`` only on a rewrite —
    so a custom endpoint must trigger one for it to take effect.
    """
    roles_cur = (current_core.get("llm") or {}).get("roles") or {}
    for role_name in ("chat", "vision", "auxiliary"):
        cur = roles_cur.get(role_name) or {}
        new = role_cfg[role_name]
        if cur.get("provider_id") != new["provider_id"]:
            return True
        if cur.get("model") != new["model"]:
            return True

    emb_new = role_cfg["embedding"]
    if (roles_cur.get("embedding") or {}).get("model") != emb_new["model"]:
        return True

    emb_endpoint_default = current_env.get("EMBEDDING_SERVER") or DEFAULT_EMBEDDING_ENDPOINT
    if emb_new["endpoint"] != emb_endpoint_default:
        return True

    memory_cur = current_core.get("memory") or {}
    cur_dim = (memory_cur.get("embeddings") or {}).get("dimension")
    if cur_dim is None or int(cur_dim) != int(emb_new["dimension"]):
        return True
    if (memory_cur.get("mem0") or {}).get("llm_provider") != provider:
        return True

    return False


def _bare_model(model: str) -> str:
    return model.removeprefix("openai/")


def _apply_llm_config(current_core: dict[str, Any], role_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``current_core`` with the wizard's LLM config applied.

    All keys written are profile-safe prefixes (``llm``, ``memory.mem0``,
    ``memory.embeddings``). The embedding endpoint is referenced via
    ``$EMBEDDING_SERVER`` (what ``get_embedding_remote_endpoint`` reads), and
    ``memory.mem0.llm_provider`` is set so only LM Studio takes its special branch.
    """
    from universal_agentic_framework.cli import steuermann

    data = copy.deepcopy(current_core)
    provider = role_cfg["chat"]["provider_id"]

    for role_name in ("chat", "vision", "auxiliary"):
        cfg = role_cfg[role_name]
        steuermann._set_dot_path(data, f"llm.roles.{role_name}.provider_id", cfg["provider_id"])
        steuermann._set_dot_path(data, f"llm.roles.{role_name}.api_base", cfg["api_base"])
        steuermann._set_dot_path(data, f"llm.roles.{role_name}.api_key", cfg["api_key"])
        steuermann._set_dot_path(data, f"llm.roles.{role_name}.model", cfg["model"])

    emb = role_cfg["embedding"]
    # Authoritative embedding fields (read by the Mem0 backend / embedding factory):
    steuermann._set_dot_path(data, "llm.roles.embedding.provider_id", provider)
    steuermann._set_dot_path(data, "llm.roles.embedding.api_base", "$EMBEDDING_SERVER")
    steuermann._set_dot_path(data, "llm.roles.embedding.model", emb["model"])
    steuermann._set_dot_path(data, "memory.embeddings.dimension", int(emb["dimension"]))
    # Cosmetic-only mirrors (kept consistent; not consumed by Mem0):
    steuermann._set_dot_path(data, "memory.embeddings.model", _bare_model(emb["model"]))
    steuermann._set_dot_path(data, "memory.embeddings.remote_endpoint", "$EMBEDDING_SERVER")
    # REQUIRED: only "lmstudio" takes the LM-Studio response-format branch.
    steuermann._set_dot_path(data, "memory.mem0.llm_provider", provider)

    return data


def write_profile_llm_config(profile_id: str, role_cfg: dict[str, Any]) -> Path | None:
    """Rewrite ``config/profiles/<id>/core.yaml`` with the wizard's LLM config.

    PyYAML ``safe_dump`` (no backup) — comments are dropped (same trade-off as
    ``config set``; the content is captured in ``docs/configuration.md``). Returns
    the written path, or ``None`` when the result is identical to the current file.
    """
    from universal_agentic_framework.cli import steuermann

    core_path = steuermann._profiles_root() / profile_id / "core.yaml"
    current = steuermann._read_yaml(core_path)
    updated = _apply_llm_config(current, role_cfg)
    if updated == current:
        return None
    core_path.write_text(
        yaml.safe_dump(updated, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return core_path


# --- Orchestrator ------------------------------------------------------------


def cmd_setup_init(args: argparse.Namespace) -> int:
    """Run the interactive first-time setup wizard."""
    from universal_agentic_framework.cli import steuermann

    fmt = args.format
    overrides: dict[str, str] = {}

    _say("🚀 Steuermann first-time setup")
    _say("   (press Enter to accept the suggested default at each step)")
    _say()

    # 1. .env guard ----------------------------------------------------------
    env_path = Path(".env")
    if env_path.exists() and not args.force:
        if _is_tty():
            if not _prompt_bool(
                f"{env_path} already exists. Overwrite? (a timestamped backup is kept)",
                default=False,
            ):
                payload = {"status": "aborted", "error": "Existing .env left unchanged"}
                steuermann._print_payload(payload, fmt)
                return 2
        else:
            payload = {
                "status": "error",
                "error": "Refusing to overwrite existing .env without --force (non-interactive)",
            }
            steuermann._print_payload(payload, fmt)
            return 2

    example_path = Path(".env.example")
    if not example_path.exists():
        payload = {"status": "error", "error": f"{example_path} not found; run from the repo root"}
        steuermann._print_payload(payload, fmt)
        return 2
    example_text = example_path.read_text(encoding="utf-8")
    current_env = _read_env_file_values(env_path)

    # 2. Platform ------------------------------------------------------------
    plat = detect_platform()
    _say(f"🖥️  Platform: {plat['os']} ({plat['machine']}), uid={plat['uid']} gid={plat['gid']}")
    if plat["is_linux"] and plat["uid"] is not None and plat["gid"] is not None:
        overrides["APP_UID"] = str(plat["uid"])
        overrides["APP_GID"] = str(plat["gid"])
    if plat["pi5_qdrant_warning"]:
        _say(
            "   ⚠️  ARM Linux detected: use the haktansuren/qdrant-pi5-fixed-jemalloc "
            "Qdrant image (see docs/configuration.md)."
        )
    _say()

    # 3. Profile (provisional) ----------------------------------------------
    provisional, available, exists, profile_error = resolve_profile(args)
    if profile_error:
        payload = {"status": "error", "error": profile_error, "available_profiles": available}
        steuermann._print_payload(payload, fmt)
        return 2
    source_profile = provisional if exists else DEFAULT_PROFILE
    current_core = steuermann._read_yaml(steuermann._profiles_root() / source_profile / "core.yaml")
    _say(f"📦 Profile: {provisional}" + ("" if exists else " (new — will scaffold from starter)"))
    _say()

    # 4. Provider + models + embedding --------------------------------------
    provider = _prompt_choice("LLM provider", PROVIDER_CHOICES, default=args.provider)
    penv = PROVIDER_ENV[provider]
    endpoint_default = current_env.get(penv["api_base_var"]) or penv["default_api_base"]
    endpoint = _prompt_with_default(f"{provider} endpoint", endpoint_default)

    if provider == "openrouter":
        key_default = args.openrouter_api_key or current_env.get(penv["api_key_var"]) or ""
        api_key = _prompt_with_default("OpenRouter API key", key_default)
        if not api_key:
            _say("   ⚠️  No OpenRouter API key provided; LLM_PROVIDERS_OPENROUTER_API_KEY left blank.")
    else:
        key_default = current_env.get(penv["api_key_var"]) or penv["default_api_key"]
        api_key = _prompt_with_default(f"{provider} API key", key_default)

    role_cfg = prompt_provider_config(
        provider, current_core, current_env, endpoint=endpoint, api_key=api_key, args=args
    )
    changed = _config_differs_from_current(current_core, role_cfg, provider, current_env)
    _say()

    # 4b. Resolve final profile ---------------------------------------------
    final_profile, scaffolded_from, scaffold_error = maybe_scaffold_for_customization(
        provisional, exists, changed, args=args
    )
    if scaffold_error:
        payload = {"status": "error", "error": scaffold_error, "available_profiles": available}
        steuermann._print_payload(payload, fmt)
        return 2
    if scaffolded_from:
        _say(
            f"🧬 Customization detected — scaffolded '{final_profile}' from "
            f"'{scaffolded_from}' (kept pristine)."
        )
    overrides["PROFILE_ID"] = final_profile
    overrides["EMBEDDING_SERVER"] = role_cfg["embedding"]["endpoint"]
    overrides.update(provider_env_overrides(provider, endpoint, api_key))

    # 5. Secrets + auth ------------------------------------------------------
    overrides["AUTH_USERNAME"] = args.auth_username
    overrides["AUTH_ADMIN_EMAIL"] = args.auth_email
    overrides["AUTH_SESSION_SECRET"] = generate_token_hex()
    overrides["CHAT_ACCESS_TOKEN"] = generate_token_hex()

    existing_pg = current_env.get("POSTGRES_PASSWORD", "").strip()
    pg_password = existing_pg if existing_pg else generate_postgres_password()
    overrides["POSTGRES_PASSWORD"] = pg_password
    pg_preserved = bool(existing_pg)

    admin_password = _prompt_password("Admin password (Enter to auto-generate)")
    if not admin_password:
        admin_password = secrets.token_urlsafe(16)
    password_hash = generate_password_hash(admin_password)

    auth_enabled = password_hash is not None
    overrides["AUTH_ENABLED"] = "true" if auth_enabled else "false"
    if auth_enabled:
        overrides["AUTH_PASSWORD_HASH"] = password_hash  # single-quoted by renderer
    else:
        _say(
            "   ⚠️  argon2-cffi unavailable — AUTH_ENABLED=false; install argon2-cffi "
            "and re-run to enable login."
        )

    # 6. Checkpointer DSN ----------------------------------------------------
    pg_user = current_env.get("POSTGRES_USER") or "framework"
    pg_host = current_env.get("POSTGRES_HOST") or "postgres"
    pg_port = current_env.get("POSTGRES_PORT") or "5432"
    pg_db = current_env.get("POSTGRES_DB") or "framework"
    overrides["CHECKPOINTER_POSTGRES_DSN"] = (
        f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    )

    # 7. Generate .env -------------------------------------------------------
    content, _extras = render_env_from_example(example_text, overrides)
    env_backup = write_env_file(env_path, content)
    _say(f"✅ Wrote {env_path}" + (f" (backup: {env_backup})" if env_backup else ""))

    # 7b. Write profile LLM config ------------------------------------------
    profile_config_path: str | None = None
    if changed:
        written = write_profile_llm_config(final_profile, role_cfg)
        if written is not None:
            profile_config_path = str(written)
            _say(f"✅ Wrote {written} (comments dropped; see docs/configuration.md)")

    # 8. Data dirs -----------------------------------------------------------
    data_dirs = ensure_data_dirs(plat["uid"], plat["gid"], is_linux=plat["is_linux"])
    if data_dirs["note"]:
        _say(f"   ⚠️  {data_dirs['note']}")

    # 9. Validation ----------------------------------------------------------
    validation = steuermann._setup_check_payload(steuermann._load_env_file())
    validation_ok = validation.get("status") != "error"

    # 10. Summary ------------------------------------------------------------
    notes: list[str] = []
    if endpoint != role_cfg["embedding"]["endpoint"]:
        notes.append(
            "LLM provider host differs from the embedding host: the Mem0 embedder reuses the "
            "auxiliary provider's API key against EMBEDDING_SERVER (local servers ignore it)."
        )
    if pg_preserved:
        notes.append("Preserved the existing POSTGRES_PASSWORD from .env (avoids breaking an initialized DB volume).")
    else:
        notes.append("Generated a new POSTGRES_PASSWORD; regenerating secrets after Postgres initialized its volume breaks DB auth.")
    if plat["pi5_qdrant_warning"]:
        notes.append("ARM Linux: use the haktansuren/qdrant-pi5-fixed-jemalloc Qdrant image.")
    notes.append("Changing the embedding dimension requires recreating existing Qdrant collections (see CLAUDE.md).")

    generated_credentials = {
        "admin_username": args.auth_username,
        "admin_password": admin_password,
        "POSTGRES_PASSWORD": pg_password,
        "AUTH_SESSION_SECRET": overrides["AUTH_SESSION_SECRET"],
        "CHAT_ACCESS_TOKEN": overrides["CHAT_ACCESS_TOKEN"],
        "AUTH_PASSWORD_HASH": "present" if auth_enabled else "absent (argon2 unavailable)",
    }

    payload = {
        "status": "ok" if validation_ok else "validation_failed",
        "platform": plat,
        "profile": final_profile,
        "scaffolded_from": scaffolded_from,
        "provider": provider,
        "models": {
            "chat": role_cfg["chat"]["model"],
            "vision": role_cfg["vision"]["model"],
            "auxiliary": role_cfg["auxiliary"]["model"],
            "embedding": {
                "endpoint": role_cfg["embedding"]["endpoint"],
                "model": role_cfg["embedding"]["model"],
                "dimension": role_cfg["embedding"]["dimension"],
            },
        },
        "env_path": str(env_path),
        "env_backup": str(env_backup) if env_backup else None,
        "profile_config_path": profile_config_path,
        "auth_enabled": auth_enabled,
        "data_dirs": data_dirs,
        "validation": validation,
        "generated_credentials": generated_credentials,
        "next_steps": [
            "docker compose up -d",
            f"Log in as '{args.auth_username}' with the generated admin password above.",
            "To disable login (dev bypass): set AUTH_ENABLED=false in .env.",
        ],
        "notes": notes,
    }

    _say()
    _say("🔑 Save these now — they are shown only once:")
    _say(f"   admin user:          {args.auth_username}")
    _say(f"   admin password:      {admin_password}")
    _say(f"   POSTGRES_PASSWORD:   {pg_password}")
    _say(f"   AUTH_SESSION_SECRET: {overrides['AUTH_SESSION_SECRET']}")
    _say(f"   CHAT_ACCESS_TOKEN:   {overrides['CHAT_ACCESS_TOKEN']}")
    _say()
    _say("Next: docker compose up -d")
    _say()

    steuermann._print_payload(payload, fmt)
    return 0 if validation_ok else 1


# --- Registration ------------------------------------------------------------


def add_setup_init_subcommand(setup_subparsers: argparse._SubParsersAction) -> None:
    """Register ``steuermann setup init`` under the ``setup`` subparser group."""
    from universal_agentic_framework.cli import steuermann

    init_parser = setup_subparsers.add_parser(
        "init", help="Interactive first-time setup wizard (writes .env, secrets, profile config)"
    )
    init_parser.add_argument("--profile", help="Target profile id (default: starter)")
    init_parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default="lmstudio",
        help="LLM provider to configure (default: lmstudio)",
    )
    init_parser.add_argument("--openrouter-api-key", default="", help="OpenRouter API key (provider=openrouter)")
    init_parser.add_argument("--auth-username", default="admin", help="Bootstrap admin username (default: admin)")
    init_parser.add_argument(
        "--auth-email", default="admin@example.com", help="Bootstrap admin email (default: admin@example.com)"
    )
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing .env without confirmation")
    steuermann._add_common_format_arg(init_parser)
    init_parser.set_defaults(func=cmd_setup_init)

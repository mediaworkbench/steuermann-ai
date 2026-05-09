# Steuermann CLI Reference

The `steuermann` CLI is the operational companion for setup validation, configuration diagnostics, profile lifecycle management, and documentation conformance checking. It does **not** replace direct YAML and `.env` editing — it supports it.

---

## Global Options

Every command accepts `--format {human,json}` (default: `human`). Use `--format json` for stable machine-readable output in CI pipelines.

Exit codes follow a strict contract:
- `0` — success (or drift detected without `--strict`)
- `1` — blocking error or validation failure
- Non-zero for all unexpected errors

---

## Command Groups

| Group | Purpose |
|---|---|
| `profile` | Inspect, scaffold, and bundle profiles |
| `config` | Show, explain, validate, and mutate configuration |
| `setup` | Run host preflight and environment checks |
| `docs` | Check documentation conformance against the contract registry |
| `ingest` | Ingest documents into RAG collections |

---

## `profile` — Profile Operations

### `steuermann profile active`

Show the resolved active profile id, directory path, and metadata validity.

```bash
steuermann profile active [--profile <id>] [--format json]
```

- `--profile` — override `PROFILE_ID` for this invocation only

**Example:**
```bash
poetry run steuermann profile active --format json
```

---

### `steuermann profile scaffold`

Create a new profile overlay directory from a source template. Copies the full required file set and writes a valid `profile.yaml` with the target profile id.

```bash
steuermann profile scaffold --from <source_profile_id> --profile <target_profile_id>
```

- `--from` — source profile to copy from (e.g. `starter`)
- `--profile` — target profile id (directory under `config/profiles/`)

**Example:**
```bash
poetry run steuermann profile scaffold \
  --from starter \
  --profile medical-ai-de
```

---

### `steuermann profile bundle export`

Package a profile directory into a portable `.tar.gz` bundle with compatibility metadata. The bundle includes the full profile overlay and a `bundle_manifest.yaml` recording the framework version range and required keys.

```bash
steuermann profile bundle export --profile <id> --out <path.tar.gz>
```

- `--profile` — profile id to export
- `--out` — output bundle path (must end in `.tar.gz`)

**Example:**
```bash
poetry run steuermann profile bundle export \
  --profile starter \
  --out /tmp/starter-bundle.tar.gz
```

---

### `steuermann profile bundle import`

Import a profile bundle into a target directory after validating compatibility metadata. Existing profiles with the same name are never overwritten — the command aborts if the target already exists.

```bash
steuermann profile bundle import --bundle <path.tar.gz> --profile <target_profile_id>
```

- `--bundle` — bundle path to import
- `--profile` — target profile id (directory under `config/profiles/`, must not already exist)

**Example:**
```bash
poetry run steuermann profile bundle import \
  --bundle /tmp/starter-bundle.tar.gz \
  --profile imported-profile
```

---

## `config` — Configuration Operations

### `steuermann config show`

Render the fully merged effective configuration for a profile (base → profile overlay → env substitutions applied).

```bash
steuermann config show [--profile <id>] [--section <section>] [--format json]
```

- `--profile` — override `PROFILE_ID`
- `--section` — filter output to a single section: `core`, `features`, `tools`, `agents`, `ui`, `profile_metadata`

**Example:**
```bash
poetry run steuermann config show --profile starter --section core --format json
```

---

### `steuermann config explain`

Show the value of a specific dot-path key and trace its provenance (which layer set it: base, profile overlay, or environment variable).

```bash
steuermann config explain --key <dot.path> [--profile <id>] [--format json]
```

- `--key` — dot-path to the key (e.g. `core.llm.providers.primary.temperature`)

**Example:**
```bash
poetry run steuermann config explain --key core.rag.collection_name --profile starter
```

---

### `steuermann config validate`

Validate the configuration for one or all profiles. Checks schema conformance, required file presence, disallowed profile overrides, invalid model string formats, and unresolved required environment substitutions.

```bash
steuermann config validate [--profile <id>] [--strict] [--format json]
```

- `--profile` — validate a specific profile (default: base + all profiles)
- `--strict` — treat warnings as failures (non-zero exit)

**Example:**
```bash
poetry run steuermann config validate --profile medical-ai-de --strict --format json
```

---

### `steuermann config set`

Set a profile-safe key in a profile's `core.yaml` overlay. Operates as a **dry-run by default** — use `--apply --confirm APPLY` to persist the change. Deployment-global keys are blocked regardless of flags.

On failure after writing, the original file is automatically restored from backup.

**Profile-safe key prefixes** (the only keys `config set` and `config unset` will accept):

| Prefix | Description |
|---|---|
| `fork.language` | Profile language code |
| `fork.locale` | Locale string |
| `fork.timezone` | Timezone |
| `fork.supported_languages` | Languages offered in the settings UI |
| `llm` | All LLM provider settings (temperature, models, timeouts, etc.) |
| `prompts` | Prompt template overrides |
| `tool_routing` | Tool routing thresholds and model |
| `rag` | RAG collection, chunk, and retrieval settings |
| `tokens` | Token budget settings |
| `memory.retention` | Memory retention periods |

All other keys are considered deployment-global and will be rejected.

```bash
steuermann config set \
  --profile <id> \
  --key <dot.path> \
  --value <yaml_value> \
  [--apply] \
  [--confirm APPLY]
```

- `--profile` — target profile (base config is not allowed)
- `--key` — dot-path key to set (must be in the profile-safe allowlist)
- `--value` — new value (parsed as a YAML scalar or object)
- `--apply` — persist the change (required alongside `--confirm APPLY`)
- `--confirm APPLY` — explicit confirmation token required for apply mode

**Example (dry-run):**
```bash
poetry run steuermann config set \
  --profile starter \
  --key core.llm.providers.primary.temperature \
  --value 0.5
```

**Example (apply):**
```bash
poetry run steuermann config set \
  --profile starter \
  --key core.llm.providers.primary.temperature \
  --value 0.5 \
  --apply --confirm APPLY
```

---

### `steuermann config unset`

Remove a key from a profile's `core.yaml` overlay, falling back to the base value at runtime. Operates as a **dry-run by default**. Same safety guardrails as `config set`.

```bash
steuermann config unset \
  --profile <id> \
  --key <dot.path> \
  [--apply] \
  [--confirm APPLY]
```

**Example:**
```bash
poetry run steuermann config unset \
  --profile starter \
  --key core.llm.providers.primary.temperature \
  --apply --confirm APPLY
```

---

### `steuermann config contract-check`

Validate parity between the CLI contract registry (`config/contracts/cli_contract.yaml`) and the runtime loader. Reports any keys, mutators, or policy clauses that are out of sync.

```bash
steuermann config contract-check [--format json]
```

**Example:**
```bash
poetry run steuermann config contract-check --format json
```

---

## `setup` — Setup Diagnostics

### `steuermann setup doctor`

Run host preflight checks: required environment variables, compose/profile alignment, profile metadata consistency, model id format validation, and collection alignment. Returns actionable remediation hints for every failed check.

Blocking checks return non-zero exit. Advisory checks emit warnings but return zero unless `--strict` is used.

```bash
steuermann setup doctor [--probe-endpoints] [--format json]
```

- `--probe-endpoints` — attempt live HTTP probes to configured LLM and embedding endpoints

**Example:**
```bash
poetry run steuermann setup doctor --probe-endpoints --format json
```

---

## `docs` — Documentation Conformance

### `steuermann docs check`

Compare repository documentation against the contract registry to detect drift: missing environment variables in docs, stale precedence rules, outdated profile-safe field listings, and setup prerequisites. Emits a structured drift report by domain. Does **not** modify any repository files.

```bash
steuermann docs check [--strict] [--format json]
```

- `--strict` — return non-zero exit if any drift is detected (suitable for CI)

**Example:**
```bash
poetry run steuermann docs check --format json
poetry run steuermann docs check --strict  # fail on drift
```

---

## `ingest` — RAG Ingestion

### `steuermann ingest ingest`

Index documents from a directory into a Qdrant collection. Skips unchanged files via content hashing.

```bash
steuermann ingest ingest --source <dir> --collection <name> [--language <lang>]
```

### `steuermann ingest watch`

Start watch mode: initial sweep + watchdog filesystem monitoring + 30-second periodic fallback sweep. Automatically re-indexes changed files and purges deleted ones.

```bash
steuermann ingest watch --source <dir> --collection <name> [--language <lang>]
```

### `steuermann ingest validate`

Validate documents without indexing them (parse-only, no Qdrant writes).

```bash
steuermann ingest validate --source <dir>
```

### `steuermann ingest reindex`

Clear an existing collection and perform a full re-index from scratch.

```bash
steuermann ingest reindex --source <dir> --collection <name> [--language <lang>]
```

---

## Typical Workflows

### New profile setup

```bash
# 1. Scaffold from starter
poetry run steuermann profile scaffold \
  --from starter \
  --profile my-profile

# 2. Edit config/profiles/my-profile/core.yaml (model, language, RAG, etc.)

# 3. Validate before activating
poetry run steuermann config validate --profile my-profile --strict --format json

# 4. Run full suite
poetry run pytest -q

# 5. Activate
PROFILE_ID=my-profile docker compose up -d --build
```

### Pre-commit check (CI-style)

```bash
poetry run steuermann config validate --strict --format json
poetry run steuermann config contract-check --format json
poetry run steuermann docs check --strict --format json
poetry run pytest -q
```

### Safe profile key override

```bash
# Dry-run first
poetry run steuermann config set \
  --profile my-profile \
  --key core.llm.providers.primary.temperature \
  --value 0.3

# Apply after review
poetry run steuermann config set \
  --profile my-profile \
  --key core.llm.providers.primary.temperature \
  --value 0.3 \
  --apply --confirm APPLY
```

### Profile portability (bundle round-trip)

```bash
# Export from source repository
poetry run steuermann profile bundle export \
  --profile starter \
  --out /tmp/starter.tar.gz

# Import into target repository
poetry run steuermann profile bundle import \
  --bundle /tmp/starter.tar.gz \
  --profile imported-starter

# Validate immediately after import
poetry run steuermann config validate --profile imported-starter --format json
```

---

## Contract Registry

The CLI contract is stored at `config/contracts/cli_contract.yaml`. It records:

- **sections** — config files and their schema metadata
- **policies** — operational validation rules
- **severity_policy** — which check failures are blocking vs advisory
- **profile_safety** — which keys are profile-safe vs deployment-global
- **profile_bundle_compatibility** — framework version range and required bundle keys
- **mutator_surface** — exact guardrail specification for `config set` and `config unset`

Use `steuermann config contract-check` to verify the registry stays in sync with the runtime loader.

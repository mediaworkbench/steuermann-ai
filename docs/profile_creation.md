# Profile Setup Guide

Use profiles to adapt the shared template to a domain without rewriting the framework.

---

## What A Profile Overlay Is

A profile overlay:

- Lives under `config/profiles/<profile_id>/`
- Is activated with `PROFILE_ID`
- Overrides base configuration, prompts, tools, and UI settings
- Can add domain-specific plugins in `plugins/`
- Should avoid direct edits in `universal_agentic_framework/core/`

---

## Minimal Workflow

```bash
cd steuermann-ai
mkdir -p config/profiles/medical-ai-de
cp config/profiles/starter/profile.yaml config/profiles/medical-ai-de/
cp config/profiles/starter/core.yaml config/profiles/medical-ai-de/
cp config/profiles/starter/features.yaml config/profiles/medical-ai-de/
cp config/profiles/starter/agents.yaml config/profiles/medical-ai-de/
cp config/profiles/starter/tools.yaml config/profiles/medical-ai-de/
cp config/profiles/starter/ui.yaml config/profiles/medical-ai-de/
cp -R config/profiles/starter/prompts config/profiles/medical-ai-de/
export PROFILE_ID=medical-ai-de
docker compose up -d --build
```

Equivalent scaffold flow with the canonical CLI:

```bash
cd steuermann-ai
poetry run steuermann profile scaffold --from starter --profile medical-ai-de
poetry run steuermann config validate --profile medical-ai-de --format json
```

---

## Required Files

Keep these files in every profile overlay:

- `config/profiles/<profile_id>/profile.yaml`
- `config/profiles/<profile_id>/core.yaml`
- `config/profiles/<profile_id>/features.yaml`
- `config/profiles/<profile_id>/agents.yaml`
- `config/profiles/<profile_id>/tools.yaml`
- `config/profiles/<profile_id>/ui.yaml`

Add `config/profiles/<profile_id>/prompts/<language>.yaml` when the profile needs prompt overrides.

---

## Common Customizations

- `core.yaml`: profile id, language defaults, model providers, RAG settings, token budgets
- `features.yaml`: feature flags such as crews, auth, analytics, or attachments
- `agents.yaml`: crew definitions and per-agent tool access
- `tools.yaml`: enable, disable, or tune tools for the profile
- `ui.yaml`: profile branding and frontend labels
- `plugins/`: domain-specific tool implementations

---

## Validation Checklist

After creating or updating a profile overlay:

1. Run `poetry run pytest -q`.
2. Run `poetry run steuermann config validate --profile <profile_id> --format json`.
3. Run `poetry run steuermann config contract-check --format json`.
4. Rebuild services with `docker compose up -d --build`.
5. Check [README.md](../README.md) startup URLs still match your local configuration.
6. Verify the frontend loads and `/metrics` remains reachable.
7. Confirm chat responses include the expected `profile_id` metadata path.

---

## Troubleshooting

If a profile overlay is not taking effect:

1. Confirm `PROFILE_ID` is present in the container environment.
2. Confirm the overlay files exist and parse as valid YAML.
3. Check FastAPI and LangGraph logs for config-loading warnings.
4. Rebuild after changing config, prompt, or plugin files.

---

## Related Docs

- **[configuration.md](configuration.md)**
- **[technical_architecture.md](technical_architecture.md)**
- **[README.md](../README.md)**

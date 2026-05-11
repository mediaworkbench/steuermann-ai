"""Model resolution and fallback logic helpers."""

from typing import Any, Optional, Tuple

from universal_agentic_framework.llm.provider_registry import normalize_model_id


def safe_get_model(config, language: str, preferred_model: Optional[str] = None):
    """Resolve model ID for a language with fallback to English and primary provider.

    Ensures that model selection follows the priority:
    1. preferred_model (if specified and valid)
    2. Model for the request language in primary provider
    3. English model in primary provider (fallback)
    4. First available model in primary provider

    Returns the resolved model ID (LiteLLM-formatted string).
    """
    if not config:
        raise ValueError("config is required")

    if not hasattr(config, "llm"):
        raise ValueError("config.llm not found")

    providers = config.llm.providers

    # primary provider (usually first in config or explicitly marked)
    primary_provider_key = getattr(config.llm, "primary_provider", "lmstudio")
    provider_config = getattr(providers, primary_provider_key, None)
    if not provider_config:
        raise ValueError(f"Primary provider '{primary_provider_key}' not configured")

    # Normalize input language
    normalized_lang = (language or "en").lower()
    if normalized_lang not in ("en", "de", "fr", "es"):
        normalized_lang = "en"

    # Preferred model check
    if preferred_model:
        candidate = normalize_model_id(preferred_model)
        return candidate

    # Language-specific model
    models = provider_config.get("models", {})
    if normalized_lang in models:
        model_id = models[normalized_lang]
        return normalize_model_id(model_id)

    # Fallback to English
    if "en" in models:
        model_id = models["en"]
        return normalize_model_id(model_id)

    # Fallback to any model in provider
    if models:
        model_id = next(iter(models.values()))
        return normalize_model_id(model_id)

    raise ValueError(f"No models configured for provider '{primary_provider_key}'")


def resolve_initial_model_metadata(config: Any, language: str, preferred_model: Optional[str]) -> Tuple[str, str]:
    """Resolve model ID and provider from config.

    Returns: (model_id, provider_id)
    """
    model_id = safe_get_model(config, language, preferred_model)

    # Extract provider from model_id (format: provider/model-name)
    if "/" in model_id:
        provider = model_id.split("/")[0].lower()
    else:
        provider = "openai"  # fallback

    # Normalize provider name to config key (e.g., "lm_studio" -> "lmstudio")
    provider_aliases = {"lm_studio": "lmstudio", "open_router": "openrouter"}
    provider_key = provider_aliases.get(provider, provider)

    return model_id, provider_key


def invoke_with_model_fallback(
    model_id: str,
    provider_id: str,
    language: str,
    config: Any,
    invoke_func,
    max_retries: int = 2,
) -> Any:
    """Invoke with a model, falling back to alternative models on failure.

    The fallback strategy prioritizes:
    1. Preferred model (if specified)
    2. Model for current language
    3. English model
    4. Any other model in primary provider
    5. Primary provider's fallback provider (if configured)

    invoke_func should accept (model_id, provider_id) and return the result
    or raise an exception on failure.
    """
    if not hasattr(config, "llm"):
        raise ValueError("config.llm not found")

    providers = config.llm.providers
    primary_provider = getattr(config.llm, "primary_provider", "lmstudio")

    attempts = 0
    last_error = None

    # Build fallback chain: try current model, then fallback provider
    fallback_providers = [provider_id]
    if provider_id != primary_provider:
        fallback_providers.append(primary_provider)

    for provider_key in fallback_providers:
        if attempts >= max_retries:
            break

        provider_config = getattr(providers, provider_key, None)
        if not provider_config:
            continue

        models = getattr(provider_config, "models", None) or {}
        model_candidates = [
            (language, models.get(language)),
            ("en", models.get("en")),
            (None, next(iter(models.values())) if models else None),
        ]

        for lang, candidate_model_id in model_candidates:
            if attempts >= max_retries:
                break
            if not candidate_model_id:
                continue

            try:
                candidate_normalized = normalize_model_id(candidate_model_id)
                result = invoke_func(candidate_normalized, provider_key)
                return result
            except Exception as e:
                attempts += 1
                last_error = e
                continue

    if last_error:
        raise last_error
    raise RuntimeError("All model fallback attempts exhausted")

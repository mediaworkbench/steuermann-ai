"""Region and country inference helpers for web search localization."""

import re
from functools import lru_cache
from typing import Dict, Optional


@lru_cache(maxsize=1)
def build_country_alias_map() -> Dict[str, str]:
    """Build a country name/alias to ISO2 mapping for region inference.

    Uses pycountry when available for broad coverage and falls back to
    curated aliases for common colloquial names and abbreviations.
    """
    aliases: Dict[str, str] = {
        "uk": "gb",
        "united kingdom": "gb",
        "great britain": "gb",
        "britain": "gb",
        "england": "gb",
        "scotland": "gb",
        "wales": "gb",
        "northern ireland": "gb",
        "usa": "us",
        "u.s.": "us",
        "u.s.a.": "us",
        "united states": "us",
        "america": "us",
        "uae": "ae",
        "south korea": "kr",
        "north korea": "kp",
        "czech republic": "cz",
        "ivory coast": "ci",
        "russia": "ru",
        "españa": "es",
        "espana": "es",
    }

    try:
        import pycountry

        for country in pycountry.countries:
            code = getattr(country, "alpha_2", "").lower()
            if not code:
                continue
            for attr in ("name", "official_name", "common_name"):
                value = getattr(country, attr, None)
                if not value:
                    continue
                aliases.setdefault(value.casefold(), code)
    except Exception:
        # Keep alias-only behavior if pycountry is unavailable.
        pass

    return aliases


def infer_country_iso2(text: str) -> Optional[str]:
    """Infer country ISO2 code from free-form text."""
    text_folded = text.casefold()
    country_alias_map = build_country_alias_map()

    # Longest match first prevents partial collisions (e.g., "united states"
    # before "states").
    for country_name in sorted(country_alias_map.keys(), key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(country_name)}(?!\w)", text_folded):
            return country_alias_map[country_name]
    return None


def region_for_country(country_iso2: str, search_language: str) -> str:
    """Convert ISO2 country + search language into a DDG region code."""
    country = country_iso2.lower()

    # DDG uses non-ISO country code for the UK.
    if country == "gb":
        return "uk-en"

    native_language_overrides = {
        "es": "es",
        "de": "de",
        "fr": "fr",
        "it": "it",
        "pt": "pt",
        "nl": "nl",
        "pl": "pl",
        "tr": "tr",
        "ru": "ru",
        "jp": "jp",
        "kr": "kr",
        "cn": "zh",
        "tw": "tzh",
        "hk": "tzh",
        "se": "sv",
        "dk": "da",
        "no": "no",
        "fi": "fi",
        "cz": "cs",
        "gr": "el",
        "il": "he",
        "ae": "ar",
        "sa": "ar",
    }

    language = native_language_overrides.get(country, (search_language or "en").lower())
    return f"{country}-{language}"

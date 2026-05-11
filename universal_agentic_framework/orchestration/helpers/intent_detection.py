"""Intent detection for tool routing and search localization."""

import re
from typing import Any, Dict


def detect_tool_routing_intents(user_msg: str, language: str) -> Dict[str, Any]:
    """Detect routing intents and query enrichments in one place.

    Keeping this logic centralized makes routing policy easier to evolve and test
    without changing execution semantics.
    """
    from .region_inference import infer_country_iso2, region_for_country
    
    def _clean_web_query(text: str) -> str:
        """Strip conversational wrappers so search terms stay semantically focused."""
        cleaned = text.strip()
        cleaned = re.sub(
            r"^\s*(?:can|could|would)\s+you\s+(?:please\s+)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*please\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*(?:search|find|look\s+up)\s+(?:the\s+)?web\s+(?:for|about)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*(?:search|find|look\s+up)\s+(?:for|about)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^\s*(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^\s*\d{1,2}\s+(?=(?:(?:latest|recent|top)\s+)*(?:news\b|results\b|articles\b|headlines\b))",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.strip(" \t\n\r?.!,;:")
        return cleaned or text.strip()

    def _infer_web_max_results(text: str, default: int = 8) -> int:
        """Infer requested number of web results (e.g., '3 latest news')."""
        lower = text.lower()

        patterns = [
            r"\b(\d{1,2})\s+(?:(?:latest|recent|top)\s+)*(?:news|results|articles|headlines)\b",
            r"\b(?:(?:latest|recent|top)\s+)+(\d{1,2})\s+(?:news|results|articles|headlines)\b",
            r"\b(?:show|give|find|get)\s+me\s+(\d{1,2})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                try:
                    value = int(match.group(1))
                    return min(max(value, 1), 10)
                except ValueError:
                    pass
        return default

    user_msg_lower = user_msg.casefold()
    routing_lang = (language or "en").lower()

    # Region mapping for web search MCP defaults
    language_map = {
        "de": ("de", "de-de"),
        "en": ("en", "us-en"),
        "fr": ("fr", "fr-fr"),
        "es": ("es", "es-es"),
    }
    search_language, search_region = language_map.get(routing_lang, ("en", "us-en"))

    # Universal country-aware override: infer country from the prompt and map to
    # a DDG region automatically instead of relying on a short hardcoded list.
    country_iso2 = infer_country_iso2(user_msg)
    if country_iso2:
        search_region = region_for_country(country_iso2, search_language)

    # Datetime: date/time patterns or keywords
    datetime_keywords = [
        "heute", "gestern", "morgen", "uhr", "time", "date", "datum",
        "geboren", "clock", "heure", "maintenant", "day", "today",
        "welcher tag", "what day", "quel jour", "current date",
        "aktuelles datum", "jetzt", "now", "how old", "age",
    ]
    mentions_datetime = bool(
        re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", user_msg_lower)
        or re.search(r"\b\d{1,2}:\d{2}\b", user_msg_lower)
        or any(re.search(rf"\b{re.escape(k)}\b", user_msg_lower) for k in datetime_keywords)
    )

    # Calculator: math expressions or calculation keywords
    mentions_calculation = bool(
        re.search(r"\b\d+\s*[\+\-\*/\^]\s*\d+", user_msg)  # 2 + 3, 10 * 5
        or re.search(r"\b(sqrt|log|sin|cos|tan|factorial)\s*\(", user_msg_lower)  # sqrt(16)
        or re.search(r"\b\d+\s*%\s*(of|von|de)\b", user_msg_lower)  # 15% of
        or re.search(r"\b(convert|umrechnen|convertir)\b.*\b(to|in|nach|en)\b", user_msg_lower)  # unit conversion
        or any(k in user_msg_lower for k in [
            "berechne", "rechne", "calculate", "compute", "wie viel ist",
            "how much is", "what is the sum", "average of", "durchschnitt",
            "prozent", "percent", "percentage",
            "quadratwurzel", "square root",
        ])
    )

    # File operations: file/directory keywords
    mentions_file_ops = bool(
        any(k in user_msg_lower for k in [
            "read file", "datei lesen", "lire fichier",
            "write file", "datei schreiben", "ecrire fichier",
            "list files", "dateien auflisten", "lister fichiers",
            "list directory", "verzeichnis", "directory listing",
            "file size", "dateigroesse", "file exists", "datei existiert",
            "file info", "dateiinfo",
        ])
    )

    # Explicit web-search intent: force/boost web search when user clearly asks for live lookup
    mentions_web_search = bool(
        re.search(r"\b(search|find|look\s*up|google|web\s*search|search\s+the\s+web)\b", user_msg_lower)
        or re.search(r"\b(im\s+web|im\s+internet|online\s+suchen|web\s+suchen)\b", user_msg_lower)
        or re.search(r"\b(recherche|rechercher|chercher\s+sur\s+le\s+web)\b", user_msg_lower)
    )

    # URL extraction: detect full URLs and bare domains (e.g., www.example.com)
    url_match = re.search(r"(https?://\S+|\b(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?\b)", user_msg)
    url_in_query = url_match.group(1) if url_match else None
    if url_in_query and not url_in_query.startswith(("http://", "https://")):
        url_in_query = f"https://{url_in_query}"

    # Meta-question detection: skip tool execution for questions about available tools
    asks_about_tools = any(
        keyword in user_msg_lower
        for keyword in [
            "welche tools", "what tools", "quels outils",
            "verfugbar", "available", "disposable",
            "zur verfugung", "tools hast du", "tools do you have",
            "funktionen hast", "funktionen available", "listing des outils",
        ]
    )

    # RAG save intent
    wants_save_to_rag = any(
        keyword in user_msg_lower
        for keyword in [
            "speicher", "speichern", "abspeichern", "ablegen", "sichern",
            "save", "store", "persist",
        ]
    )

    # Finance sentiment intent: enrich search query for better relevance
    sentiment_keywords = ["sentiment", "bullish", "bearish", "market mood", "social sentiment", "stimmung", "marktstimmung"]
    has_sentiment_intent = any(k in user_msg_lower for k in sentiment_keywords)
    ticker_match = re.search(r"\b[A-Z]{2,6}\b", user_msg)
    ticker = ticker_match.group(0) if ticker_match else None
    enhanced_web_query = _clean_web_query(user_msg)
    requested_web_results = _infer_web_max_results(user_msg, default=8)
    if has_sentiment_intent and ticker:
        enhanced_web_query = f"{ticker} coin crypto sentiment market outlook social media news"

    return {
        "user_msg_lower": user_msg_lower,
        "search_language": search_language,
        "search_region": search_region,
        "mentions_datetime": mentions_datetime,
        "mentions_calculation": mentions_calculation,
        "mentions_file_ops": mentions_file_ops,
        "mentions_web_search": mentions_web_search,
        "url_in_query": url_in_query,
        "asks_about_tools": asks_about_tools,
        "wants_save_to_rag": wants_save_to_rag,
        "enhanced_web_query": enhanced_web_query,
        "requested_web_results": requested_web_results,
    }

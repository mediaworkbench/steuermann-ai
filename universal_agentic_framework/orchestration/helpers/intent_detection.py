"""Intent detection for tool routing and search localization."""

import re
from typing import Any, Dict

# Queries shorter than this character count are candidates for the trivial-query RAG skip
# heuristic. Tune here if false positives emerge — do not inline this threshold.
_RAG_SKIP_SHORT_QUERY_CHARS: int = 35


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

    # Explicit web-search intent: require unambiguous "search the web" phrasing.
    # Bare "search" or "find" are too common in non-web contexts (code search, file search).
    mentions_web_search = bool(
        re.search(r"\b(search\s+(?:the\s+)?(?:web|internet)|web\s*search|look\s*up|google|search\s+for)\b", user_msg_lower)
        or re.search(r"\b(im\s+web|im\s+internet|online\s+suchen|web\s+suchen)\b", user_msg_lower)
        or re.search(r"\b(recherche|rechercher|chercher\s+sur\s+le\s+web)\b", user_msg_lower)
    )

    # URL extraction: require full URL scheme, explicit www. prefix, or a path segment on a
    # bare domain to avoid false positives from file extensions, version strings, and emails.
    url_match = re.search(
        r"(https?://\S+|www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?|\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,}/\S+)",
        user_msg,
    )
    url_in_query = url_match.group(1) if url_match else None
    if url_in_query and not url_in_query.startswith(("http://", "https://")):
        url_in_query = f"https://{url_in_query}"

    # Image URL: http(s) URL ending with an image file extension
    image_url_in_query = bool(
        re.search(r"https?://\S+\.(?:jpg|jpeg|png|gif|webp)\b", user_msg, re.IGNORECASE)
    )

    # Convenience flag used by vision-tool compound boosts (image URL or attachment)
    image_in_query = image_url_in_query

    # OCR: user wants text extracted from an image
    mentions_ocr = bool(
        re.search(r"\bocr\b", user_msg_lower)
        or re.search(
            r"\b(read|extract|transcribe|lese|lies|extrahier|Text erkennen|Text lesen|was steht)\b.*\b(text|writing|schrift|beschriftung)\b",
            user_msg_lower,
        )
        or any(k in user_msg_lower for k in [
            "what does it say", "what does the text say", "read the text",
            "extract text", "text in the image", "text im bild",
        ])
    )

    # Document analysis: user wants structured data from a document image
    mentions_document = bool(
        re.search(r"\b(invoice|receipt|form|contract|rechnung|quittung|beleg|formular|vertrag)\b", user_msg_lower)
        or any(k in user_msg_lower for k in [
            "scan document", "digitize document", "extract from document",
            "document data", "bill data", "Dokument scannen", "Beleg auslesen",
            "Formular digitalisieren", "Rechnungsdaten",
        ])
    )

    # Chart analysis: user wants data extracted from a chart or graph
    mentions_chart = bool(
        re.search(
            r"\b(chart|graph|plot|trend|visualization|diagramm|grafik|kurve|balkendiagramm|liniendiagramm|kreisdiagramm)\b",
            user_msg_lower,
        )
        or any(k in user_msg_lower for k in [
            "bar chart", "line graph", "pie chart", "scatter plot", "histogram",
            "what does the chart show", "extract data from chart",
            "was zeigt das diagramm", "Daten aus dem Diagramm",
        ])
    )

    # Image metadata / EXIF: user wants file metadata, not visual analysis
    mentions_image_metadata = bool(
        re.search(r"\b(exif|metadata|metadaten)\b", user_msg_lower)
        or any(k in user_msg_lower for k in [
            "when was this photo taken", "where was this taken", "what camera",
            "what resolution", "gps location of photo", "photo metadata",
            "wann wurde das foto", "welche kamera", "bildgröße",
        ])
    )

    # Barcode / QR code: user wants to decode a barcode or QR code
    mentions_barcode = bool(
        re.search(r"\b(barcode|qr.?code|qr code|bar code)\b", user_msg_lower)
        or any(k in user_msg_lower for k in [
            "scan code", "scan this code", "read barcode", "decode qr",
            "product code", "qr-code scannen", "barcode lesen", "code auslesen",
        ])
    )

    # CSV / spreadsheet analysis: user wants to compute over tabular data
    mentions_csv_analysis = bool(
        re.search(
            r"\b(sum|total|average|mean|count|min|max|group\s*by|filter|value.?count|head|tail|unique)\b",
            user_msg_lower,
        )
        or any(k in user_msg_lower for k in [
            # English
            "how many rows", "how many entries", "how many records",
            "sum the", "total the", "average of", "mean of",
            "aggregate", "pivot", "sort by column",
            # German
            "summe", "gesamtbetrag", "durchschnitt", "mittelwert",
            "wie viele zeilen", "wie viele einträge", "einträge zählen",
            "spalte auswerten", "csv auswerten", "tabelle auswerten",
            "gruppieren", "nach spalte", "zeilen filtern",
        ])
    )

    # Map / geocoding: user wants to see a location or measure distance
    mentions_map = bool(
        re.search(r"\b(locate|karte)\b", user_msg_lower)
        or any(k in user_msg_lower for k in [
            # English
            "where is", "where are", "map of", "show me the map",
            "show on map", "find on map", "on the map",
            "show me on", "put on the map",
            "how far", "distance from", "distance between",
            # German
            "wo ist", "wo liegt", "wo befindet", "karte von",
            "zeig auf der karte", "zeig mir auf der karte", "zeig mir die karte",
            "wie weit", "entfernung", "abstand zwischen",
        ])
    )

    # Weather / forecast: user wants current weather, a temperature comparison, or a forecast.
    # Includes comparatives ("how much warmer is A than B") so the compare prompt routes and is
    # not mistaken for a calculator query. Substring checks mirror mentions_map (EN + DE).
    mentions_weather = bool(
        re.search(r"\b(weather|wetter|temperature|temperatur|forecast|vorhersage)\b", user_msg_lower)
        or any(k in user_msg_lower for k in [
            # English
            "weather in", "weather for", "weather like", "how warm", "how cold", "how hot",
            "warmer", "colder", "hotter", "how much warmer", "how much colder",
            "current temperature", "will it rain", "going to rain", "rain this week",
            # German
            "wetter in", "wetter für", "wie ist das wetter", "wie warm", "wie kalt",
            "wärmer", "kälter", "heißer", "wie viel wärmer", "wie viel kälter",
            "aktuelle temperatur", "regnet es", "wird es regnen",
        ])
    )

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

    # RAG short-circuit: skip Qdrant entirely for trivial queries that cannot benefit
    # from knowledge retrieval regardless of what the knowledge base contains.
    _short = len(user_msg_lower.strip()) < _RAG_SKIP_SHORT_QUERY_CHARS
    _is_greeting = bool(re.search(
        r"^(hello|hi|hey|hallo|guten\s*tag|guten\s*morgen|bonjour|salut|ciao|yo)\b",
        user_msg_lower.strip(),
    ))
    skip_rag = bool(
        _is_greeting
        or (_short and mentions_calculation and not mentions_web_search)
        or (_short and mentions_datetime and not mentions_web_search)
        or asks_about_tools
    )

    # Force-tool-use: structured-mode models must not be allowed to opt out when
    # the user has clearly requested a web search.
    force_tool_use = bool(mentions_web_search)

    return {
        "user_msg_lower": user_msg_lower,
        "search_language": search_language,
        "search_region": search_region,
        "mentions_datetime": mentions_datetime,
        "mentions_calculation": mentions_calculation,
        "mentions_web_search": mentions_web_search,
        "url_in_query": url_in_query,
        "image_url_in_query": image_url_in_query,
        "image_in_query": image_in_query,
        "mentions_ocr": mentions_ocr,
        "mentions_document": mentions_document,
        "mentions_chart": mentions_chart,
        "mentions_image_metadata": mentions_image_metadata,
        "mentions_barcode": mentions_barcode,
        "mentions_map": mentions_map,
        "mentions_weather": mentions_weather,
        "mentions_csv_analysis": mentions_csv_analysis,
        "asks_about_tools": asks_about_tools,
        "wants_save_to_rag": wants_save_to_rag,
        "enhanced_web_query": enhanced_web_query,
        "requested_web_results": requested_web_results,
        "skip_rag": skip_rag,
        "force_tool_use": force_tool_use,
    }

"""Weather tool — current conditions, two-place comparison, and multi-day forecast via Open-Meteo.

Mirrors ``map_tool``: an external HTTP API wrapped as a LangChain ``BaseTool`` that returns a
JSON-stringified payload. The payload is parsed into ``.data`` downstream (by
``normalize_tool_payload`` via ``ast.literal_eval``) and surfaced to the frontend as the
``weather_data`` artifact, so the payload must contain only strings/ints/floats — never
booleans or ``None`` (those break ``ast.literal_eval`` and the widget would never render).
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Literal, Optional

import httpx
import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = structlog.get_logger()

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HEADERS = {"User-Agent": "steuermann-ai/1.0"}
_GEOCODE_PARAMS = {"count": "5", "language": "en", "format": "json"}
_CURRENT_VARS = "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation"
_DAILY_VARS = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"
_TIMEOUT = 10.0
_MAX_FORECAST_DAYS = 7

# WMO weather interpretation codes → short English condition text. The frontend widget
# localizes and picks an icon from the raw ``weather_code``; this string is the fallback
# and is what the answering model reads in the tool output.
_WMO_CONDITIONS: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherInput(BaseModel):
    """Input schema for weather operations."""

    operation: Literal["current", "compare", "forecast"] = Field(
        default="current",
        description="Operation: 'current' weather for one place, 'compare' two places, or multi-day 'forecast'.",
    )
    location: Optional[str] = Field(
        default=None,
        description="Place name for 'current' or 'forecast' (city, optionally with country, e.g. 'Barcelona, Spain').",
    )
    location_a: Optional[str] = Field(
        default=None,
        description="First place for 'compare' (e.g. 'Barcelona, Spain').",
    )
    location_b: Optional[str] = Field(
        default=None,
        description="Second place for 'compare' (e.g. 'Schwerin, Germany').",
    )
    days: Optional[int] = Field(
        default=5,
        description="Number of forecast days for 'forecast' (1-7).",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(value: Any, default: float = 0.0, ndigits: int = 1) -> float:
    """Coerce a value to a rounded float, never returning ``None`` (ast.literal_eval-safe)."""
    if value is None:
        return default
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return default


def _parse_location_query(raw: str) -> tuple[str, Optional[str]]:
    """Split 'Barcelona, Spain' / 'Barcelona (Spain)' into (name, country_hint)."""
    s = (raw or "").strip()
    paren = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", s)
    if paren:
        return paren.group(1).strip(), paren.group(2).strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[-1]
    return s, None


def _pick_geocode_result(results: list, country_hint: Optional[str]) -> dict:
    """Prefer the result whose country matches the hint; otherwise the top hit."""
    if country_hint:
        h = country_hint.casefold()
        for r in results:
            country = str(r.get("country", "")).casefold()
            code = str(r.get("country_code", "")).casefold()
            if h == country or h == code or (len(h) > 2 and h in country):
                return r
    return results[0]


def _geocode_sync(query: str) -> dict:
    """Return the best Open-Meteo geocoding result, or raise ValueError on failure."""
    name, hint = _parse_location_query(query)
    resp = httpx.get(
        _GEOCODE_URL,
        params={**_GEOCODE_PARAMS, "name": name},
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise ValueError(f"Location not found: {query!r}")
    return _pick_geocode_result(results, hint)


async def _geocode_async(query: str) -> dict:
    """Async variant using httpx.AsyncClient."""
    name, hint = _parse_location_query(query)
    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as client:
        resp = await client.get(_GEOCODE_URL, params={**_GEOCODE_PARAMS, "name": name})
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            raise ValueError(f"Location not found: {query!r}")
        return _pick_geocode_result(results, hint)


def _location_from_geocode(r: dict, query: str) -> dict:
    """Convert a raw geocoding result to a simple location dict (no None values)."""
    name = r.get("name") or query
    country = r.get("country") or ""
    label = f"{name}, {country}" if country else str(name)
    return {
        "label": label,
        "lat": float(r["latitude"]),
        "lon": float(r["longitude"]),
        "country": str(country),
        "timezone": str(r.get("timezone") or ""),
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class WeatherTool(BaseTool):
    """Current weather, two-place comparison, and multi-day forecast via Open-Meteo."""

    name: str = "weather_tool"
    description: str = (
        "Get the current weather for a place, compare the weather of two places, or get a "
        "multi-day forecast. Use when the user asks: how is the weather in [city], what's the "
        "weather in [place], current temperature in [city], how warm/cold is it in [place], "
        "how much warmer/colder is [A] than [B], compare the weather in [A] and [B], weather "
        "forecast for [city], will it rain this week. Returns temperature, conditions, humidity, "
        "wind, and an interactive weather widget."
    )
    args_schema: type[BaseModel] = WeatherInput

    # Injected by the registry from the profile's tools.yaml `config:` block (defaults here).
    # Passed straight through to Open-Meteo, which converts server-side.
    temperature_unit: str = "celsius"
    wind_speed_unit: str = "kmh"
    precipitation_unit: str = "mm"

    # ------------------------------------------------------------------
    # Sync / async entry points
    # ------------------------------------------------------------------

    def _run(
        self,
        operation: str = "current",
        location: Optional[str] = None,
        location_a: Optional[str] = None,
        location_b: Optional[str] = None,
        days: Optional[int] = 5,
    ) -> str:
        try:
            if operation == "current":
                return self._current_sync(location or "")
            elif operation == "compare":
                return self._compare_sync(location_a or "", location_b or "")
            elif operation == "forecast":
                return self._forecast_sync(location or "", days)
            else:
                return f"Error: Unknown operation '{operation}'. Use 'current', 'compare', or 'forecast'."
        except ValueError as exc:
            logger.warning("weather_tool lookup failed", error=str(exc), operation=operation)
            return f"Error: {exc}"
        except Exception as exc:
            logger.error("weather_tool unexpected error", error=str(exc), operation=operation)
            return f"Error: Could not complete weather request — {exc}"

    async def _arun(
        self,
        operation: str = "current",
        location: Optional[str] = None,
        location_a: Optional[str] = None,
        location_b: Optional[str] = None,
        days: Optional[int] = 5,
    ) -> str:
        try:
            if operation == "current":
                return await self._current_async(location or "")
            elif operation == "compare":
                return await self._compare_async(location_a or "", location_b or "")
            elif operation == "forecast":
                return await self._forecast_async(location or "", days)
            else:
                return f"Error: Unknown operation '{operation}'. Use 'current', 'compare', or 'forecast'."
        except ValueError as exc:
            logger.warning("weather_tool lookup failed", error=str(exc), operation=operation)
            return f"Error: {exc}"
        except Exception as exc:
            logger.error("weather_tool unexpected error", error=str(exc), operation=operation)
            return f"Error: Could not complete weather request — {exc}"

    # ------------------------------------------------------------------
    # Open-Meteo forecast fetch (units come from injected config)
    # ------------------------------------------------------------------

    def _forecast_params(self, lat: float, lon: float, *, daily: bool, days: int) -> Dict[str, str]:
        params = {
            "latitude": f"{lat}",
            "longitude": f"{lon}",
            "timezone": "auto",
            "temperature_unit": self.temperature_unit,
            "wind_speed_unit": self.wind_speed_unit,
            "precipitation_unit": self.precipitation_unit,
            "current": _CURRENT_VARS,
        }
        if daily:
            params["daily"] = _DAILY_VARS
            params["forecast_days"] = str(days)
        return params

    def _fetch_forecast_sync(self, lat: float, lon: float, *, daily: bool, days: int) -> dict:
        resp = httpx.get(
            _FORECAST_URL,
            params=self._forecast_params(lat, lon, daily=daily, days=days),
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    async def _fetch_forecast_async(self, lat: float, lon: float, *, daily: bool, days: int) -> dict:
        async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as client:
            resp = await client.get(
                _FORECAST_URL,
                params=self._forecast_params(lat, lon, daily=daily, days=days),
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Current — sync/async
    # ------------------------------------------------------------------

    def _current_sync(self, query: str) -> str:
        if not query.strip():
            return "Error: No location provided."
        loc = _location_from_geocode(_geocode_sync(query), query)
        fc = self._fetch_forecast_sync(loc["lat"], loc["lon"], daily=False, days=1)
        return self._build_current_payload(loc, fc)

    async def _current_async(self, query: str) -> str:
        if not query.strip():
            return "Error: No location provided."
        loc = _location_from_geocode(await _geocode_async(query), query)
        fc = await self._fetch_forecast_async(loc["lat"], loc["lon"], daily=False, days=1)
        return self._build_current_payload(loc, fc)

    # ------------------------------------------------------------------
    # Compare — sync/async (two locations fetched concurrently in async)
    # ------------------------------------------------------------------

    def _compare_sync(self, q_a: str, q_b: str) -> str:
        if not q_a.strip() or not q_b.strip():
            return "Error: Both 'location_a' and 'location_b' are required for compare."
        reading_a = self._reading_sync(q_a)
        reading_b = self._reading_sync(q_b)
        return self._build_compare_payload(reading_a, reading_b)

    async def _compare_async(self, q_a: str, q_b: str) -> str:
        if not q_a.strip() or not q_b.strip():
            return "Error: Both 'location_a' and 'location_b' are required for compare."
        reading_a, reading_b = await asyncio.gather(
            self._reading_async(q_a), self._reading_async(q_b)
        )
        return self._build_compare_payload(reading_a, reading_b)

    def _reading_sync(self, query: str) -> dict:
        loc = _location_from_geocode(_geocode_sync(query), query)
        fc = self._fetch_forecast_sync(loc["lat"], loc["lon"], daily=False, days=1)
        return self._build_reading(loc, fc)

    async def _reading_async(self, query: str) -> dict:
        loc = _location_from_geocode(await _geocode_async(query), query)
        fc = await self._fetch_forecast_async(loc["lat"], loc["lon"], daily=False, days=1)
        return self._build_reading(loc, fc)

    # ------------------------------------------------------------------
    # Forecast — sync/async
    # ------------------------------------------------------------------

    def _forecast_sync(self, query: str, days: Optional[int]) -> str:
        if not query.strip():
            return "Error: No location provided."
        n = self._clamp_days(days)
        loc = _location_from_geocode(_geocode_sync(query), query)
        fc = self._fetch_forecast_sync(loc["lat"], loc["lon"], daily=True, days=n)
        return self._build_forecast_payload(loc, fc)

    async def _forecast_async(self, query: str, days: Optional[int]) -> str:
        if not query.strip():
            return "Error: No location provided."
        n = self._clamp_days(days)
        loc = _location_from_geocode(await _geocode_async(query), query)
        fc = await self._fetch_forecast_async(loc["lat"], loc["lon"], daily=True, days=n)
        return self._build_forecast_payload(loc, fc)

    @staticmethod
    def _clamp_days(days: Optional[int]) -> int:
        try:
            n = int(days) if days is not None else 5
        except (TypeError, ValueError):
            n = 5
        return max(1, min(_MAX_FORECAST_DAYS, n))

    # ------------------------------------------------------------------
    # Payload builders (return JSON strings; values are str/int/float only)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reading(loc: dict, fc: dict) -> dict:
        """One place's current conditions — the shared block for 'current' and 'compare'."""
        current = fc.get("current") or {}
        units = fc.get("current_units") or {}
        code = int(_num(current.get("weather_code"), ndigits=0))
        return {
            "label": loc["label"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "country": loc["country"],
            "timezone": loc["timezone"],
            "temperature": _num(current.get("temperature_2m")),
            "apparent_temperature": _num(current.get("apparent_temperature")),
            "weather_code": code,
            "condition": _WMO_CONDITIONS.get(code, "Unknown"),
            "humidity_pct": int(_num(current.get("relative_humidity_2m"), ndigits=0)),
            "wind_speed": _num(current.get("wind_speed_10m")),
            "precipitation": _num(current.get("precipitation")),
            "temperature_unit": str(units.get("temperature_2m") or "°C"),
            "wind_speed_unit": str(units.get("wind_speed_10m") or "km/h"),
            "precipitation_unit": str(units.get("precipitation") or "mm"),
            "observed_at": str(current.get("time") or ""),
        }

    @classmethod
    def _build_current_payload(cls, loc: dict, fc: dict) -> str:
        reading = cls._build_reading(loc, fc)
        tu = reading["temperature_unit"]
        summary = (
            f"{reading['label']}: {reading['temperature']:.0f}{tu} "
            f"(feels like {reading['apparent_temperature']:.0f}{tu}), "
            f"{reading['condition'].lower()}, humidity {reading['humidity_pct']}%, "
            f"wind {reading['wind_speed']:.0f} {reading['wind_speed_unit']}."
        )
        payload = {"type": "current", "reading": reading, "summary": summary}
        logger.info("weather_tool current", label=reading["label"], temp=reading["temperature"])
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _build_compare_payload(a: dict, b: dict) -> str:
        tu = a["temperature_unit"]
        delta = round(a["temperature"] - b["temperature"], 1)
        if delta > 0:
            warmer = a["label"]
            relation = f"{a['label']} is {abs(delta):.0f}{tu} warmer than {b['label']}"
        elif delta < 0:
            warmer = b["label"]
            relation = f"{b['label']} is {abs(delta):.0f}{tu} warmer than {a['label']}"
        else:
            warmer = ""
            relation = f"{a['label']} and {b['label']} are the same temperature"
        summary = (
            f"{relation} "
            f"({a['temperature']:.0f}{tu} vs {b['temperature']:.0f}{tu})."
        )
        payload = {
            "type": "compare",
            "readings": [a, b],
            "delta": abs(delta),
            "delta_unit": tu,
            "warmer": warmer,
            "summary": summary,
        }
        logger.info("weather_tool compare", a=a["label"], b=b["label"], delta=delta)
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _build_forecast_payload(loc: dict, fc: dict) -> str:
        daily = fc.get("daily") or {}
        units = fc.get("daily_units") or {}
        tu = str(units.get("temperature_2m_max") or "°C")
        pu = str(units.get("precipitation_sum") or "mm")
        times = daily.get("time") or []
        codes = daily.get("weather_code") or []
        tmax = daily.get("temperature_2m_max") or []
        tmin = daily.get("temperature_2m_min") or []
        precip = daily.get("precipitation_sum") or []
        days: List[dict] = []
        for i, date in enumerate(times):
            code = int(_num(codes[i] if i < len(codes) else 0, ndigits=0))
            days.append({
                "date": str(date),
                "weather_code": code,
                "condition": _WMO_CONDITIONS.get(code, "Unknown"),
                "temp_max": _num(tmax[i] if i < len(tmax) else None),
                "temp_min": _num(tmin[i] if i < len(tmin) else None),
                "precipitation": _num(precip[i] if i < len(precip) else None),
                "temperature_unit": tu,
                "precipitation_unit": pu,
            })
        head = "; ".join(
            f"{d['date']} {d['temp_min']:.0f}–{d['temp_max']:.0f}{tu} {d['condition'].lower()}"
            for d in days[:3]
        )
        summary = f"{len(days)}-day forecast for {loc['label']}: {head}."
        payload = {
            "type": "forecast",
            "location": {
                "label": loc["label"],
                "lat": loc["lat"],
                "lon": loc["lon"],
                "country": loc["country"],
                "timezone": loc["timezone"],
            },
            "days": days,
            "summary": summary,
        }
        logger.info("weather_tool forecast", label=loc["label"], days=len(days))
        return json.dumps(payload, ensure_ascii=False)

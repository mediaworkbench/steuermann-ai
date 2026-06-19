"""Tests for weather_tool (Open-Meteo).

No respx/pytest-httpx dependency in this repo — HTTP is mocked with unittest.mock by
patching the tool module's `httpx.get` (sync) and the async forecast/geocode helpers.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from universal_agentic_framework.orchestration.helpers.intent_detection import (
    detect_tool_routing_intents,
)
from universal_agentic_framework.tools.weather.tool import (
    _WMO_CONDITIONS,
    WeatherInput,
    WeatherTool,
    _num,
    _parse_location_query,
    _pick_geocode_result,
)

_HTTPX_GET = "universal_agentic_framework.tools.weather.tool.httpx.get"

_GEO_BCN = {"name": "Barcelona", "latitude": 41.39, "longitude": 2.16, "country": "Spain", "country_code": "ES", "timezone": "Europe/Madrid"}
_GEO_SCHWERIN = {"name": "Schwerin", "latitude": 53.63, "longitude": 11.41, "country": "Germany", "country_code": "DE", "timezone": "Europe/Berlin"}


def _current_forecast(temp, unit="°C"):
    return {
        "current": {
            "time": "2026-06-18T14:00",
            "temperature_2m": temp,
            "apparent_temperature": temp - 1,
            "relative_humidity_2m": 60,
            "weather_code": 2,
            "wind_speed_10m": 12.0,
            "precipitation": 0.0,
        },
        "current_units": {"temperature_2m": unit, "wind_speed_10m": "km/h", "precipitation": "mm"},
    }


_DAILY_FORECAST = {
    "daily": {
        "time": ["2026-06-18", "2026-06-19", "2026-06-20"],
        "weather_code": [2, 61, 0],
        "temperature_2m_max": [24.0, 19.0, 26.0],
        "temperature_2m_min": [15.0, 12.0, 16.0],
        "precipitation_sum": [0.0, 4.2, 0.0],
    },
    "daily_units": {"temperature_2m_max": "°C", "precipitation_sum": "mm"},
}


def _fake_get_factory(*, temp_unit="°C"):
    """Return a side_effect that answers geocode + forecast by URL/params."""
    def _fake_get(url, params=None, headers=None, timeout=None):
        resp = Mock()
        resp.raise_for_status = Mock()
        params = params or {}
        if "geocoding-api" in url:
            name = str(params.get("name", "")).lower()
            result = _GEO_SCHWERIN if "schwerin" in name else _GEO_BCN
            resp.json = Mock(return_value={"results": [result]})
        elif params.get("daily"):
            resp.json = Mock(return_value=_DAILY_FORECAST)
        else:
            lat = str(params.get("latitude", ""))
            temp = 13.0 if lat.startswith("53") else 25.0
            resp.json = Mock(return_value=_current_forecast(temp, unit=temp_unit))
        return resp
    return _fake_get


def _assert_no_bool_or_none(obj):
    """ast.literal_eval (downstream) rejects JSON true/false/null — payload must avoid them."""
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_bool_or_none(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_bool_or_none(v)
    else:
        assert obj is not None
        assert not isinstance(obj, bool)


# ── Input schema & helpers ───────────────────────────────────────────────

def test_input_defaults():
    inp = WeatherInput()
    assert inp.operation == "current"
    assert inp.days == 5


def test_parse_location_query():
    assert _parse_location_query("Barcelona, Spain") == ("Barcelona", "Spain")
    assert _parse_location_query("Barcelona (Spain)") == ("Barcelona", "Spain")
    assert _parse_location_query("Berlin") == ("Berlin", None)


def test_pick_geocode_result_prefers_country_hint():
    results = [_GEO_BCN, {**_GEO_SCHWERIN, "name": "Barcelona"}]
    # hint Germany should pick the second (country Germany) even though it's not first
    picked = _pick_geocode_result(results, "Germany")
    assert picked["country"] == "Germany"
    # no hint → first
    assert _pick_geocode_result(results, None) is results[0]


def test_num_coalesces_none():
    assert _num(None) == 0.0
    assert _num("nan-ish") == 0.0
    assert _num(12.345) == 12.3


def test_wmo_map():
    assert _WMO_CONDITIONS[0] == "Clear sky"
    assert _WMO_CONDITIONS[95].startswith("Thunderstorm")


# ── current ──────────────────────────────────────────────────────────────

def test_current_returns_widget_payload():
    with patch(_HTTPX_GET, side_effect=_fake_get_factory()):
        out = WeatherTool()._run(operation="current", location="Barcelona, Spain")
    data = json.loads(out)
    assert data["type"] == "current"
    assert data["reading"]["label"] == "Barcelona, Spain"
    assert data["reading"]["temperature"] == 25.0
    assert data["reading"]["temperature_unit"] == "°C"
    assert data["reading"]["condition"] == "Partly cloudy"
    assert "Barcelona" in data["summary"]
    _assert_no_bool_or_none(data)


def test_current_missing_location():
    assert WeatherTool()._run(operation="current", location="").startswith("Error")


def test_location_not_found():
    def _empty(url, params=None, headers=None, timeout=None):
        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json = Mock(return_value={"results": []})
        return resp

    with patch(_HTTPX_GET, side_effect=_empty):
        out = WeatherTool()._run(operation="current", location="Nowhereville")
    assert out.startswith("Error: Location not found")


# ── compare ────────────────────────────────────────────────────────────────

def test_compare_delta_and_warmer():
    with patch(_HTTPX_GET, side_effect=_fake_get_factory()):
        out = WeatherTool()._run(
            operation="compare", location_a="Barcelona, Spain", location_b="Schwerin, Germany"
        )
    data = json.loads(out)
    assert data["type"] == "compare"
    assert len(data["readings"]) == 2
    assert data["delta"] == 12.0           # 25 - 13
    assert data["warmer"] == "Barcelona, Spain"
    assert "warmer" in data["summary"]
    _assert_no_bool_or_none(data)


def test_compare_requires_both():
    assert WeatherTool()._run(operation="compare", location_a="Berlin").startswith("Error")


# ── forecast ────────────────────────────────────────────────────────────────

def test_forecast_days():
    with patch(_HTTPX_GET, side_effect=_fake_get_factory()):
        out = WeatherTool()._run(operation="forecast", location="Berlin", days=3)
    data = json.loads(out)
    assert data["type"] == "forecast"
    assert len(data["days"]) == 3
    assert data["days"][1]["condition"] == "Slight rain"   # code 61
    assert data["days"][0]["temp_max"] == 24.0
    _assert_no_bool_or_none(data)


def test_forecast_days_clamped():
    assert WeatherTool._clamp_days(99) == 7
    assert WeatherTool._clamp_days(0) == 1
    assert WeatherTool._clamp_days(None) == 5


# ── unit config (Celsius / Fahrenheit) ──────────────────────────────────────

def test_fahrenheit_config_passes_param_and_labels():
    captured = {}

    def _fake_get(url, params=None, headers=None, timeout=None):
        resp = Mock()
        resp.raise_for_status = Mock()
        params = params or {}
        if "geocoding-api" in url:
            resp.json = Mock(return_value={"results": [_GEO_BCN]})
        else:
            captured.update(params)
            resp.json = Mock(return_value=_current_forecast(77.0, unit="°F"))
        return resp

    with patch(_HTTPX_GET, side_effect=_fake_get):
        out = WeatherTool(temperature_unit="fahrenheit")._run(
            operation="current", location="Barcelona, Spain"
        )
    data = json.loads(out)
    assert captured["temperature_unit"] == "fahrenheit"
    assert data["reading"]["temperature_unit"] == "°F"
    assert "°F" in data["summary"]


# ── async path (compare uses asyncio.gather) ─────────────────────────────────

@pytest.mark.asyncio
async def test_async_compare():
    geo = AsyncMock(side_effect=[_GEO_BCN, _GEO_SCHWERIN])
    fc = AsyncMock(side_effect=[_current_forecast(25.0), _current_forecast(13.0)])
    with patch("universal_agentic_framework.tools.weather.tool._geocode_async", geo), \
         patch.object(WeatherTool, "_fetch_forecast_async", fc):
        out = await WeatherTool()._arun(
            operation="compare", location_a="Barcelona, Spain", location_b="Schwerin, Germany"
        )
    data = json.loads(out)
    assert data["type"] == "compare"
    assert data["delta"] == 12.0


# ── intent detection routes both example prompts ─────────────────────────────

@pytest.mark.parametrize("prompt", [
    "how is the weather in barcelona, spain?",
    "how much warmer is barcelona (spain) than schwerin (germany)?",
    "wetter in Berlin diese Woche",
])
def test_weather_intent_detected(prompt):
    intents = detect_tool_routing_intents(prompt, "en")
    assert intents["mentions_weather"] is True

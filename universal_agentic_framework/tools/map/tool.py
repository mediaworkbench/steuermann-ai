"""Map tool — geocode locations and compute distances via Nominatim (OSM)."""

import json
import math
from typing import List, Literal, Optional

import httpx
import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = structlog.get_logger()

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_HEADERS = {"User-Agent": "steuermann-ai/1.0"}
_NOMINATIM_PARAMS = {"format": "json", "limit": "1", "addressdetails": "0"}
_TIMEOUT = 10.0


class MapInput(BaseModel):
    """Input schema for map operations."""

    operation: Literal["locate", "distance", "multi"] = Field(
        default="locate",
        description="Operation: 'locate' a place, compute 'distance' between two places, or 'multi' for multiple pins.",
    )
    location: Optional[str] = Field(
        default=None,
        description="Place name for 'locate' (city, country, region, landmark).",
    )
    from_location: Optional[str] = Field(
        default=None,
        description="Starting place for 'distance'.",
    )
    to_location: Optional[str] = Field(
        default=None,
        description="Destination place for 'distance'.",
    )
    locations: Optional[List[str]] = Field(
        default=None,
        description="List of place names for 'multi' (e.g. ['Berlin', 'Paris', 'Rome']).",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zoom_from_bbox(bbox: list) -> int:
    """Derive a sensible map zoom level from a Nominatim bounding box."""
    span = max(float(bbox[1]) - float(bbox[0]), float(bbox[3]) - float(bbox[2]))
    if span < 0.1:
        return 14   # street / neighbourhood
    if span < 1:
        return 12   # city
    if span < 5:
        return 10   # metro area / province
    if span < 20:
        return 7    # small country
    if span < 60:
        return 5    # large country / subcontinent
    return 3        # continent / world


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _geocode_sync(query: str) -> dict:
    """Return Nominatim result dict or raise ValueError on failure."""
    resp = httpx.get(
        _NOMINATIM_URL,
        params={**_NOMINATIM_PARAMS, "q": query},
        headers=_NOMINATIM_HEADERS,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Location not found: {query!r}")
    return results[0]


async def _geocode_async(query: str) -> dict:
    """Async variant using httpx.AsyncClient."""
    async with httpx.AsyncClient(headers=_NOMINATIM_HEADERS, timeout=_TIMEOUT) as client:
        resp = await client.get(_NOMINATIM_URL, params={**_NOMINATIM_PARAMS, "q": query})
        resp.raise_for_status()
        results = resp.json()
        if not results:
            raise ValueError(f"Location not found: {query!r}")
        return results[0]


def _result_to_location(r: dict, query: str) -> dict:
    """Convert a raw Nominatim result to a simple location dict."""
    return {
        "label": r.get("display_name", query),
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
    }


def _osm_url_locate(lat: float, lon: float, zoom: int) -> str:
    return f"https://www.openstreetmap.org/#map={zoom}/{lat:.6f}/{lon:.6f}"


def _osm_url_directions(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    return f"https://www.openstreetmap.org/directions?from={lat1:.6f},{lon1:.6f}&to={lat2:.6f},{lon2:.6f}"


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class MapTool(BaseTool):
    """Geocode locations and compute distances using OpenStreetMap / Nominatim."""

    name: str = "map_tool"
    description: str = (
        "Show a location, region, or country on an interactive map, or measure the "
        "straight-line distance between two places. Use when the user asks: where is "
        "[city/place/country], show me [location], show me the map of [region/continent], "
        "map of Europe, where is Berlin, locate [place], how far is [A] from [B], "
        "distance between [A] and [B], how many kilometers from [A] to [B], "
        "show me Berlin and Paris and Rome. Always use this tool when the user wants "
        "to see a location visually on a map."
    )
    args_schema: type[BaseModel] = MapInput

    # ------------------------------------------------------------------
    # Sync entry point
    # ------------------------------------------------------------------

    def _run(
        self,
        operation: str = "locate",
        location: Optional[str] = None,
        from_location: Optional[str] = None,
        to_location: Optional[str] = None,
        locations: Optional[List[str]] = None,
    ) -> str:
        try:
            if operation == "locate":
                return self._locate_sync(location or "")
            elif operation == "distance":
                return self._distance_sync(from_location or "", to_location or "")
            elif operation == "multi":
                return self._multi_sync(locations or [])
            else:
                return f"Error: Unknown operation '{operation}'. Use 'locate', 'distance', or 'multi'."
        except ValueError as exc:
            logger.warning("map_tool geocoding failed", error=str(exc), operation=operation)
            return f"Error: {exc}"
        except Exception as exc:
            logger.error("map_tool unexpected error", error=str(exc), operation=operation)
            return f"Error: Could not complete map request — {exc}"

    # ------------------------------------------------------------------
    # Async entry point
    # ------------------------------------------------------------------

    async def _arun(
        self,
        operation: str = "locate",
        location: Optional[str] = None,
        from_location: Optional[str] = None,
        to_location: Optional[str] = None,
        locations: Optional[List[str]] = None,
    ) -> str:
        try:
            if operation == "locate":
                return await self._locate_async(location or "")
            elif operation == "distance":
                return await self._distance_async(from_location or "", to_location or "")
            elif operation == "multi":
                return await self._multi_async(locations or [])
            else:
                return f"Error: Unknown operation '{operation}'. Use 'locate', 'distance', or 'multi'."
        except ValueError as exc:
            logger.warning("map_tool geocoding failed", error=str(exc), operation=operation)
            return f"Error: {exc}"
        except Exception as exc:
            logger.error("map_tool unexpected error", error=str(exc), operation=operation)
            return f"Error: Could not complete map request — {exc}"

    # ------------------------------------------------------------------
    # Locate — sync/async
    # ------------------------------------------------------------------

    def _locate_sync(self, query: str) -> str:
        if not query.strip():
            return "Error: No location provided."
        r = _geocode_sync(query)
        return self._build_locate_payload(r, query)

    async def _locate_async(self, query: str) -> str:
        if not query.strip():
            return "Error: No location provided."
        r = await _geocode_async(query)
        return self._build_locate_payload(r, query)

    @staticmethod
    def _build_locate_payload(r: dict, query: str) -> str:
        lat = float(r["lat"])
        lon = float(r["lon"])
        label = r.get("display_name", query)
        zoom = _zoom_from_bbox(r["boundingbox"]) if r.get("boundingbox") else 12
        lat_str = f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}"
        lon_str = f"{abs(lon):.4f}°{'E' if lon >= 0 else 'W'}"
        payload = {
            "type": "location",
            "label": label,
            "lat": lat,
            "lon": lon,
            "zoom": zoom,
            "osm_url": _osm_url_locate(lat, lon, zoom),
            "summary": f"{label} is located at {lat_str}, {lon_str}.",
        }
        logger.info("map_tool locate", label=label, lat=lat, lon=lon, zoom=zoom)
        return json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Distance — sync/async
    # ------------------------------------------------------------------

    def _distance_sync(self, from_q: str, to_q: str) -> str:
        if not from_q.strip() or not to_q.strip():
            return "Error: Both 'from_location' and 'to_location' are required for distance."
        r_from = _geocode_sync(from_q)
        r_to = _geocode_sync(to_q)
        return self._build_distance_payload(r_from, from_q, r_to, to_q)

    async def _distance_async(self, from_q: str, to_q: str) -> str:
        if not from_q.strip() or not to_q.strip():
            return "Error: Both 'from_location' and 'to_location' are required for distance."
        r_from = await _geocode_async(from_q)
        r_to = await _geocode_async(to_q)
        return self._build_distance_payload(r_from, from_q, r_to, to_q)

    @staticmethod
    def _build_distance_payload(r_from: dict, from_q: str, r_to: dict, to_q: str) -> str:
        loc_a = _result_to_location(r_from, from_q)
        loc_b = _result_to_location(r_to, to_q)
        dist_km = _haversine_km(loc_a["lat"], loc_a["lon"], loc_b["lat"], loc_b["lon"])
        dist_mi = dist_km * 0.621371
        midpoint = {
            "lat": (loc_a["lat"] + loc_b["lat"]) / 2,
            "lon": (loc_a["lon"] + loc_b["lon"]) / 2,
        }
        zoom = _zoom_from_bbox([
            min(loc_a["lat"], loc_b["lat"]),
            max(loc_a["lat"], loc_b["lat"]),
            min(loc_a["lon"], loc_b["lon"]),
            max(loc_a["lon"], loc_b["lon"]),
        ])
        # Zoom out one level so both pins have breathing room
        zoom = max(2, zoom - 1)
        payload = {
            "type": "distance",
            "distance_km": round(dist_km, 1),
            "distance_miles": round(dist_mi, 1),
            "locations": [loc_a, loc_b],
            "midpoint": {"lat": round(midpoint["lat"], 4), "lon": round(midpoint["lon"], 4)},
            "zoom": zoom,
            "osm_url": _osm_url_directions(loc_a["lat"], loc_a["lon"], loc_b["lat"], loc_b["lon"]),
            "summary": (
                f"The straight-line distance from {loc_a['label']} to {loc_b['label']} "
                f"is {dist_km:.0f} km ({dist_mi:.0f} miles)."
            ),
        }
        logger.info(
            "map_tool distance",
            from_label=loc_a["label"],
            to_label=loc_b["label"],
            dist_km=round(dist_km, 1),
        )
        return json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Multi — sync/async
    # ------------------------------------------------------------------

    def _multi_sync(self, queries: List[str]) -> str:
        if not queries:
            return "Error: No locations provided for 'multi'."
        results = [_geocode_sync(q) for q in queries]
        return self._build_multi_payload(results, queries)

    async def _multi_async(self, queries: List[str]) -> str:
        if not queries:
            return "Error: No locations provided for 'multi'."
        results = []
        for q in queries:
            results.append(await _geocode_async(q))
        return self._build_multi_payload(results, queries)

    @staticmethod
    def _build_multi_payload(results: list, queries: List[str]) -> str:
        locations = [_result_to_location(r, q) for r, q in zip(results, queries)]
        lats = [loc["lat"] for loc in locations]
        lons = [loc["lon"] for loc in locations]
        bbox = [min(lats), max(lats), min(lons), max(lons)]
        zoom = max(2, _zoom_from_bbox(bbox) - 1)
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        labels = "; ".join(loc["label"].split(",")[0] for loc in locations)
        payload = {
            "type": "multi",
            "locations": locations,
            "zoom": zoom,
            "osm_url": _osm_url_locate(center_lat, center_lon, zoom),
            "summary": f"Showing {labels} on the map.",
        }
        logger.info("map_tool multi", count=len(locations))
        return json.dumps(payload, ensure_ascii=False)

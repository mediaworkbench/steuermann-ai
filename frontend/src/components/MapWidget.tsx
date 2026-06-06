"use client";

import { useEffect, useRef } from "react";
import type { StyleSpecification } from "maplibre-gl";
import type { MapData } from "@/lib/types";

// CSS is imported globally in layout.tsx — MapLibre must not import it here
// because component-level CSS imports are unreliable in Next.js production builds.

const OPENFREEMAP_STYLE = "https://tiles.openfreemap.org/styles/positron";
const SHOW_PIN_THRESHOLD = 5; // skip marker for zoom ≤ this (continent/world view)
const DISTANCE_LINE_SOURCE_ID = "distance-line";
const DISTANCE_LINE_LAYER_ID = "distance-line";
const MAP_COLOR_PRIMARY = "#2563eb";
const MAP_COLOR_SECONDARY = "#dc2626";
const MAP_COLOR_DISTANCE_LINE = "#6366f1";
const MAP_COLOR_SUCCESS = "#16a34a";
const MAP_COLOR_WARNING = "#d97706";
const MAP_COLOR_ACCENT = "#7c3aed";
const MAP_MULTI_MARKER_COLORS = [
  MAP_COLOR_PRIMARY,
  MAP_COLOR_SECONDARY,
  MAP_COLOR_SUCCESS,
  MAP_COLOR_WARNING,
  MAP_COLOR_ACCENT,
];

// Module-level cache so repeated mounts (multiple map results in history) share
// one fetch instead of re-requesting the style JSON each time.
let cachedStyle: StyleSpecification | null = null;

// MapLibre v4.x throws when a numeric comparison operator receives null from a
// tile feature property. Wrap bare `["get", "prop"]` expressions inside ordered
// comparisons with `["coalesce", ["get", "prop"], 0]` so null falls through as 0.
const NUMERIC_CMP_OPS = new Set([">=", "<=", ">", "<"]);
function nullSafeFilter(expr: unknown): unknown {
  if (!Array.isArray(expr)) return expr;
  const [op, ...args] = expr;
  if (
    typeof op === "string" &&
    NUMERIC_CMP_OPS.has(op) &&
    Array.isArray(args[0]) &&
    args[0][0] === "get"
  ) {
    return [op, ["coalesce", args[0], 0], ...args.slice(1)];
  }
  return [op, ...args.map(nullSafeFilter)];
}

interface Props {
  data: MapData;
}

export function MapWidget({ data }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let cancelled = false;
    let mapInstance: import("maplibre-gl").Map | null = null;

    (async () => {
      try {
        const maplibregl = await import("maplibre-gl");

        if (cancelled || !containerRef.current) return;

        const center: [number, number] =
          data.type === "location" && data.lat != null && data.lon != null
            ? [data.lon, data.lat]
            : data.type === "distance" && data.midpoint
            ? [data.midpoint.lon, data.midpoint.lat]
            : data.locations && data.locations.length > 0
            ? [
                data.locations.reduce((s, l) => s + l.lon, 0) / data.locations.length,
                data.locations.reduce((s, l) => s + l.lat, 0) / data.locations.length,
              ]
            : [0, 0];

        // Fetch and sanitize the style for MapLibre v4.x compatibility:
        // 1. Strip terrain / raster-dem sources (pre-existing guard).
        // 2. Wrap numeric comparison `get` expressions with `coalesce` so that
        //    features whose properties are null don't throw "Expected value to be
        //    of type number, but found null instead" in the tile-parsing worker.
        //    MapLibre v3.x silently coerced nulls; v4.x is strict.
        if (!cachedStyle) {
          const styleResp = await fetch(OPENFREEMAP_STYLE);
          const styleJson = await styleResp.json();
          delete styleJson.terrain;
          if (styleJson.sources) {
            for (const key of Object.keys(styleJson.sources)) {
              if (styleJson.sources[key].type === "raster-dem") {
                delete styleJson.sources[key];
              }
            }
          }
          if (styleJson.layers) {
            styleJson.layers = styleJson.layers.map(
              (layer: { filter?: unknown }) =>
                layer.filter
                  ? { ...layer, filter: nullSafeFilter(layer.filter) }
                  : layer
            );
          }
          cachedStyle = styleJson as StyleSpecification;
        }

        // Check again — component may have unmounted during the style fetch.
        if (cancelled || !containerRef.current) return;

        mapInstance = new maplibregl.Map({
          container: containerRef.current,
          style: cachedStyle,
          center,
          zoom: data.zoom,
          scrollZoom: false,
          boxZoom: false,
          doubleClickZoom: false,
          dragRotate: false,
          touchZoomRotate: false,
          attributionControl: false,
        });

        // Attribution at bottom-right (compact) — keeps bottom-left clear for the distance badge.
        mapInstance.addControl(
          new maplibregl.AttributionControl({ compact: true }),
          "bottom-right"
        );

        // Provide a transparent 1×1 placeholder for any missing sprite image so
        // the tile pipeline never stalls. Triggered e.g. by "road_" when a road
        // feature has null ref_length (["concat","road_",["get","ref_length"]]).
        mapInstance.on("styleimagemissing", (e: { id: string }) => {
          if (!mapInstance?.hasImage(e.id)) {
            mapInstance?.addImage(e.id, {
              width: 1,
              height: 1,
              data: new Uint8Array(4),
            });
          }
        });

        mapInstance.on("load", () => {
          if (cancelled || !mapInstance) return;

          if (
            data.type === "location" &&
            data.lat != null &&
            data.lon != null &&
            data.zoom > SHOW_PIN_THRESHOLD
          ) {
            new maplibregl.Marker({ color: MAP_COLOR_PRIMARY })
              .setLngLat([data.lon, data.lat])
              .setPopup(new maplibregl.Popup({ closeButton: false }).setText(data.label ?? ""))
              .addTo(mapInstance);
          }

          if (data.type === "distance" && data.locations && data.locations.length === 2) {
            const [a, b] = data.locations;

            new maplibregl.Marker({ color: MAP_COLOR_PRIMARY })
              .setLngLat([a.lon, a.lat])
              .setPopup(new maplibregl.Popup({ closeButton: false }).setText(a.label))
              .addTo(mapInstance);

            new maplibregl.Marker({ color: MAP_COLOR_SECONDARY })
              .setLngLat([b.lon, b.lat])
              .setPopup(new maplibregl.Popup({ closeButton: false }).setText(b.label))
              .addTo(mapInstance);

            if (mapInstance.getLayer(DISTANCE_LINE_LAYER_ID)) {
              mapInstance.removeLayer(DISTANCE_LINE_LAYER_ID);
            }
            if (mapInstance.getSource(DISTANCE_LINE_SOURCE_ID)) {
              mapInstance.removeSource(DISTANCE_LINE_SOURCE_ID);
            }

            mapInstance.addSource(DISTANCE_LINE_SOURCE_ID, {
              type: "geojson",
              data: {
                type: "Feature",
                geometry: {
                  type: "LineString",
                  coordinates: [
                    [a.lon, a.lat],
                    [b.lon, b.lat],
                  ],
                },
                properties: {},
              },
            });
            mapInstance.addLayer({
              id: DISTANCE_LINE_LAYER_ID,
              type: "line",
              source: DISTANCE_LINE_SOURCE_ID,
              layout: { "line-join": "round", "line-cap": "round" },
              paint: {
                "line-color": MAP_COLOR_DISTANCE_LINE,
                "line-width": 2,
                "line-dasharray": [4, 3],
              },
            });

            mapInstance.fitBounds(
              [
                [Math.min(a.lon, b.lon), Math.min(a.lat, b.lat)],
                [Math.max(a.lon, b.lon), Math.max(a.lat, b.lat)],
              ],
              { padding: 32, animate: false }
            );
          }

          if (data.type === "multi" && data.locations && data.locations.length > 0) {
            data.locations.forEach((loc, i) => {
              new maplibregl.Marker({ color: MAP_MULTI_MARKER_COLORS[i % MAP_MULTI_MARKER_COLORS.length] })
                .setLngLat([loc.lon, loc.lat])
                .setPopup(new maplibregl.Popup({ closeButton: false }).setText(loc.label))
                .addTo(mapInstance!);
            });

            const lons = data.locations.map((l) => l.lon);
            const lats = data.locations.map((l) => l.lat);
            mapInstance.fitBounds(
              [
                [Math.min(...lons), Math.min(...lats)],
                [Math.max(...lons), Math.max(...lats)],
              ],
              { padding: 32, animate: false }
            );
          }
        });
      } catch {
        // Style fetch or MapLibre init failed — container stays blank.
        mapInstance?.remove();
        mapInstance = null;
      }
    })();

    return () => {
      cancelled = true;
      mapInstance?.remove();
    };
  }, [data]);

  return (
    <div
      className="relative w-full max-w-sm rounded-lg border border-border shadow-sm"
      style={{ height: "192px" }}
    >
      <div ref={containerRef} className="rounded-lg overflow-hidden" style={{ width: "100%", height: "100%" }} />

      {data.type === "distance" && data.distance_km != null && (
        <div className="absolute bottom-6 left-2 rounded bg-foreground/60 px-2 py-0.5 text-xs font-medium text-background">
          {data.distance_km} km · {data.distance_miles ?? "?"} mi
        </div>
      )}

      <a
        href={data.osm_url}
        target="_blank"
        rel="noopener noreferrer"
        className="absolute top-1 right-1 rounded bg-foreground/60 px-2 py-0.5 text-xs font-medium text-background hover:bg-foreground/75"
      >
        Open full map ↗
      </a>
    </div>
  );
}

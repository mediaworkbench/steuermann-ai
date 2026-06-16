# Tool Catalog

This page is the user-facing reference for every tool built into Steuermann. For instructions on building your own tools, see [tool_development_guide.md](tool_development_guide.md).

---

## How Tools Are Loaded

Tools are registered in `config/profiles/<profile_id>/tools.yaml` and loaded at graph startup via the tool registry. Which tools are active for a given request is determined by a three-tier selection process:

1. **Layer 1 — Semantic prefilter:** Each query is embedded and scored via cosine similarity against tool descriptions. Intent boosts (+0.2) raise the score for tools that match recognized patterns (see the table below). Only tools above `similarity_threshold: 0.55` and passing the spread gate pass through.
2. **Layer 2 — LLM-driven selection:** The language model receives the candidate tools and decides which to call, in `native`, `structured`, or `react` mode depending on the model's capabilities.
3. **Layer 3 — Validation:** Tool name and arguments are validated against schema. Failed calls are retried up to 2 times with error feedback.

Which tools each user may use is controlled by role — see [Controlling Tool Access](#controlling-tool-access).

For the full selection architecture, see [tool_development_guide.md](tool_development_guide.md) § Tool Selection Architecture.

---

## Quick Reference

| Tool | Category | Vision LLM required | Intent boost |
| --- | --- | --- | --- |
| `calculator_tool` | Utility | No | Yes — math expressions, `berechne`, `calculate`, `sqrt(...)` |
| `datetime_tool` | Utility | No | Yes — date/time patterns, `heute`, `time`, `date` |
| `map_tool` | Utility | No | Yes — "where is", "map of", "how far", "distance between", "locate" |
| `file_ops_tool` | Utility | No | No |
| `web_search_mcp` | Network (MCP) | No | No direct boost — URL detection boosts content extraction routing |
| `analyze_image_tool` | Vision | **Yes** | Yes — image URL (`.jpg/.png/.gif/.webp`) in message or image attachment in state |
| `ocr_tool` | Vision | **Yes** | Yes — "read text", "OCR", "Text erkennen" |
| `analyze_document_tool` | Vision | **Yes** | Yes — "scan invoice", "Rechnung scannen", "analyse document" |
| `analyze_chart_tool` | Vision | **Yes** | Yes — "analyze chart", "Diagramm", "chart" |
| `image_metadata_tool` | Image Library | No | No |
| `read_barcodes_tool` | Image Library | No | Yes — "scan barcode", "QR-Code", "barcode" |
| `csv_analyze_tool` | Spreadsheet | No | Yes — "sum/total/average/count/group by/filter" on a CSV workspace doc |

---

## Utility Tools

These tools are always available regardless of the active LLM model and require no external services.

### `calculator_tool`

Evaluates arithmetic and mathematical expressions.

**Capabilities:**

- Basic arithmetic, exponentiation, and parenthesized expressions
- Percentage calculations (`15% of 200`)
- Trigonometric functions (`sin`, `cos`, `tan`, `atan2`)
- Square root and logarithms (`sqrt(144)`, `log(1000)`)
- Unit conversions (kilometers to miles, Celsius to Fahrenheit, etc.)
- Statistical functions on lists (mean, median, standard deviation)
- Multilingual trigger recognition: German (`berechne`, `rechne`, `wie viel ist`), French (`calcule`, `combien`), English

**Returns:** Computed numeric result with optional unit label.

**Intent boost triggers:** Numeric expressions with operators (`*`, `/`, `+`, `-`, `^`, `%`), `sqrt(`, `log(`, `calculate`, `berechne`, `calcule`.

**Configuration:** No profile-level config knobs. Enabled/disabled via `config/profiles/<id>/tools.yaml`.

---

### `datetime_tool`

Returns the current date and time, optionally in a specified timezone.

**Capabilities:**

- Current date and time in any IANA timezone (e.g. `Europe/Berlin`, `America/New_York`)
- Formatted output adapting to the user's language
- Date arithmetic (days until an event, days since a date)
- Day-of-week lookups

**Returns:** Formatted date/time string with timezone label.

**Intent boost triggers:** `today`, `heute`, `what time`, `what day`, `current date`, `now`, `jetzt`, `uhrzeit`, `datum`.

**Configuration key:** `default_timezone` in `config/profiles/<profile_id>/core.yaml` (defaults to `UTC` if not set).

---

### `map_tool`

Geocodes locations and measures straight-line distances using [Nominatim](https://nominatim.openstreetmap.org/) (OpenStreetMap) — free, no API key required. Results are displayed as an interactive map widget in the chat (MapLibre GL JS + [OpenFreeMap](https://openfreemap.org/) tiles, also free). Clicking the widget opens a full-resolution OpenStreetMap view in a new tab.

**Operations:**

| Operation | Description | Example trigger |
| --- | --- | --- |
| `locate` | Geocode a city, country, region, or continent; auto-zoom from bounding box | "Where is Berlin?", "Show me the map of Europe" |
| `distance` | Haversine straight-line distance between two places; two-pin map with dashed line | "How far is London from Madrid?" |
| `multi` | Multiple pins on one map with `fitBounds` | "Show me Berlin, Paris, and Rome" |

**Auto-zoom logic:** Zoom level is derived from the Nominatim bounding box — city → 12, metro area → 10, country → 6, continent → 4. A marker pin is omitted for zoom ≤ 5 (continent and world views have no meaningful single point).

**Returns:** A JSON string with structured fields (`type`, `lat`, `lon`, `zoom`, `osm_url`, `summary`, and operation-specific extras). The `summary` field contains a human-readable sentence for the LLM; coordinates are consumed by the `MapWidget` frontend component.

**Intent boost triggers:** "where is", "where are", "map of", "show me the map", "how far", "distance from", "distance between", "locate", "wo ist", "wo liegt", "karte von", "wie weit", "entfernung".

**No configuration required** — Nominatim is a public endpoint (rate-limit: 1 req/s; acceptable for AI chat). No API keys, no Docker service.

---

### `file_ops_tool`

Reads and writes files within a sandboxed workspace directory.

**Capabilities:**

- Read file contents by filename or relative path
- Write or overwrite a file
- List files in the workspace
- Operations are sandboxed to the configured workspace root — paths outside are rejected

**Returns:** File content (for reads), success confirmation (for writes), or directory listing.

**Intent boost triggers:** None — selected via LLM judgment when file-related intent is present in the message.

**Configuration keys** in `config/profiles/<profile_id>/core.yaml`:

| Key | Default | Description |
| --- | --- | --- |
| `sandbox_dir` | `/tmp/steuermann-ai` | Absolute path of the sandboxed workspace |
| `max_read_size_bytes` | `1048576` (1 MB) | Maximum file size for read operations |
| `allowed_extensions` | `[.txt, .md, .json, .yaml, .csv]` | File extensions the tool will read/write |

---

## Spreadsheet Tools

### `csv_analyze_tool`

Computes exact aggregates over workspace CSV/spreadsheet files using Python's stdlib `csv` module — no pandas required. Operates over the **full file** on disk, bypassing the LLM context window limit.

**Operations:**

| Operation | Description | Example trigger |
| --- | --- | --- |
| `summary` | Row count, column names, inferred numeric vs text per column | "Describe this spreadsheet", "what columns are there?" |
| `aggregate` | `sum` / `mean` / `min` / `max` / `count` over a column; optional `group_by` | "sum the amount column", "average price per category" |
| `filter` | Rows where `column <op> value`; ops: `== != > >= < <= contains` | "filter where status == active" |
| `head` / `tail` | First / last N rows (default 20) | "show the first 10 rows" |
| `unique` | Distinct values in a column | "unique values in the status column" |
| `value_counts` | Top-N value frequencies | "most common categories" |

**Delimiter detection:** `csv.Sniffer` automatically detects `,` `;` `\t` `|`  — handles German `;`-delimited exports. Falls back to `,`.

**Numeric coercion:** Tolerant of `1.234,56` (DE) and `1,234.56` (EN) formats, plus leading currency symbols (`€`, `$`, `£`).

**File access:** The tool receives the CSV's absolute `stored_path` via the structured tool-calling node's system prompt (C0 injection). It validates the path is inside the workspace root (`CHAT_WORKSPACE_ROOT`) before reading — paths outside are rejected. **Structured mode only** in v1 (native/react don't inject the path; known limitation).

**Intent boost triggers (gated on a `text/csv` workspace doc being present):** "sum", "total", "average", "mean", "count", "group by", "filter", "how many rows", "how many entries", "value counts", "head", "tail", "unique", "Summe", "Durchschnitt", "Mittelwert", "wie viele Zeilen", "Spalte auswerten", "CSV auswerten", "Tabelle auswerten", "gruppieren", "filtern".

**Returns:** Compact human-readable text (not JSON). Errors start with `"Error: …"`.

**No configuration required** — reads from the same workspace root as all other workspace documents.

---

## Network / MCP Tools

These tools communicate with external services over the Docker network.

### `web_search_mcp`

Searches the web via DuckDuckGo and optionally fetches page content. Runs as a Model Context Protocol (MCP) server inside the Docker stack (`duckduckgo-mcp:8000`).

**Capabilities:**

- Keyword and natural-language web search via DuckDuckGo
- Fetches and extracts text content from a URL
- Region-configurable results (default: `de-de`)

**Exposed operations:** `search` (returns titles, snippets, URLs) and `fetch_content` (returns extracted page text for a given URL).

**URL intent routing:** When a URL is detected anywhere in the user's message, the routing layer applies a +0.2 intent boost to content-extraction routing. This is handled generically by URL pattern detection — not tied to a single tool — so the model can use `web_search_mcp` to fetch the page.

**Returns:** Search results as a structured list of (title, snippet, URL) tuples, or page content as raw text.

**Configuration keys** in the tool manifest (`config/profiles/<id>/tools.yaml`):

| Key | Default | Description |
| --- | --- | --- |
| `server_url` | `http://duckduckgo-mcp:8000/mcp` | MCP server endpoint |
| `region` | `de-de` | BCP-47 region tag for DuckDuckGo results |
| `max_results` | `5` | Maximum search results returned |

> **Note:** The `duckduckgo-mcp` container must be healthy for this tool to function. Check with `docker compose logs duckduckgo-mcp` if search fails.

---

## Vision Tools

These four tools send images to the vision-capable LLM role. They all require `llm.roles.vision` to be configured in the active profile overlay. If `vision` is not present, the runtime falls back to the `chat` role — which may not support image input.

All vision tools share:

- Maximum image size: **10 MB**
- Attachment resolution: images referenced by path are loaded from `attachments_base_dir` (configurable per profile)
- Shared helpers in `universal_agentic_framework/tools/vision_utils.py` (base64 encoding, MIME detection, size validation)

### `analyze_image_tool`

General-purpose image understanding and description.

**Capabilities:**

- Describe image contents in natural language
- Answer specific questions about what is shown in an image
- Identify objects, scenes, text visible in the image, and layout

**Input:** Image URL (direct link or file path) or an image attachment already uploaded to the conversation.

**Returns:** Natural-language description or answer to the posed question.

**Intent boost triggers:** Image URL pattern in message (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`), or image attachment present in conversation state.

---

### `ocr_tool`

Extracts text from images using the vision LLM.

**Capabilities:**

- Reads printed and handwritten text from photos, screenshots, and scans
- Preserves layout and line structure where possible
- Multilingual — recognizes any language visible in the image

**Returns:** Extracted text as a plain string, preserving line breaks.

**Intent boost triggers:** `ocr`, `read text`, `text erkennen`, `extract text`, `was steht`, `text aus bild`, `read the text in`.

---

### `analyze_document_tool`

Analyzes scanned documents, invoices, forms, and structured paperwork.

**Capabilities:**

- Extracts structured fields from invoices, receipts, and forms (vendor, date, line items, totals)
- Summarizes document content
- Identifies document type and key data points

**Returns:** Structured JSON with extracted fields, or a natural-language summary depending on the prompt.

**Intent boost triggers:** `scan invoice`, `Rechnung scannen`, `analyse document`, `document analysis`, `Dokument analysieren`, `receipt`, `form`.

---

### `analyze_chart_tool`

Interprets charts, graphs, and data visualizations.

**Capabilities:**

- Identifies chart type (bar, line, pie, scatter, etc.)
- Extracts data trends, peaks, and anomalies described in the chart
- Reads axis labels, legend entries, and titles

**Returns:** Natural-language analysis of the chart, including extracted values where readable.

**Intent boost triggers:** `analyze chart`, `Diagramm`, `chart analysis`, `graph`, `Diagramm analysieren`, `what does this chart show`.

---

## Image Library Tools

These tools process images using local libraries (Pillow, pyzbar) and do **not** send images to the LLM. They work regardless of the active LLM model configuration.

### `image_metadata_tool`

Extracts EXIF metadata and technical properties from image files using Pillow.

**Capabilities:**

- Image format, dimensions (width × height), color mode, and DPI
- Camera make and model (from EXIF if present)
- Capture date and time
- GPS coordinates (latitude/longitude) if embedded
- Embedded copyright and artist fields

**Returns:** Structured dictionary of metadata fields, omitting absent fields.

**Intent boost triggers:** None — selected via LLM judgment when metadata extraction intent is present.

---

### `read_barcodes_tool`

Decodes barcodes and QR codes from images using pyzbar.

**Capabilities:**

- QR codes, Code 128, Code 39, EAN-13, EAN-8, UPC-A, UPC-E, Data Matrix, and other formats supported by pyzbar
- Returns decoded data, barcode type, and bounding-box position
- Multiple barcodes in a single image are all decoded

**Returns:** List of decoded results: `{data, type, position}` per barcode found.

**Intent boost triggers:** `scan barcode`, `QR-Code`, `barcode`, `QR code`, `read barcode`, `decode`.

> **System dependency:** pyzbar requires `libzbar0` (Linux) or `zbar` (macOS). This is installed in the Docker image. If running tools directly on the host, install the system library before running tests that exercise this tool.

---

## Controlling Tool Access

A profile's `config/profiles/<profile_id>/tools.yaml` is the **catalog** — the complete set of tools loaded for that profile. There is no per-tool `enabled` flag; every listed tool is loaded. (To stop a tool from loading at all, remove its entry from `tools.yaml`.) Profile-specific `config` overrides are still applied per entry:

```yaml
# config/profiles/my-profile/tools.yaml
tools:
  - name: web_search_mcp
    path: universal_agentic_framework/tools/web_search
    config:
      region: en-us       # profile-specific config override
      max_results: 10
```

**Who may use each tool is decided at runtime, not in config:**

1. **Per role (admin-controlled).** Administrators assign the allowed tools for each role (`user`, `researcher`) on the **Admin** page. The mapping is stored in the `role_tool_permissions` table and enforced **server-side** in the graph's `load_tools` node — a user can never invoke a tool their role isn't allowed, even if the UI is bypassed. Administrators always have access to every tool. A role with **no** stored row blocks all tools (fail-closed); on a fresh deployment `user` and `researcher` are seeded with the full catalog.
2. **Per user — saved preference (Settings).** Each user enables/disables their allowed tools in **Settings** (persisted as `tool_toggles`). This is the user's standing preference and controls **which tools appear in the chat composer's Tools menu** — a tool turned off in Settings is hidden from the menu. A toggle can never re-enable a tool the role disallows.
3. **Per user — per-chat quick toggle (composer).** The composer's **Tools** menu is a transient, per-conversation quick disable: clicking a tool turns it off for *that chat's* next inferences only (it stays listed, shown OFF, and can be re-enabled). This never changes the saved Settings preference. The disabled set is sent with each chat request (`disabled_tools`) and overlaid onto `tool_toggles` for that inference only.

On the Admin and Settings pages tools are grouped into three columns — **Text**, **Vision**, and **Auxiliary** — derived from each tool's manifest `category` field.

The profile's `tools.yaml` is the sole source of truth for the catalog — there is no base registry to inherit from.

---

## Related Documentation

- **[tool_development_guide.md](tool_development_guide.md)** — building custom tools, MCP integrations, tool manifest schema
- **[configuration.md](configuration.md)** — full configuration reference including `tools.yaml` schema
- **[deployment_guide.md](deployment_guide.md)** — Docker service topology (where `duckduckgo-mcp` runs)

## Plan: Frontend Role Surfaces

Design and implement a clean role-oriented frontend with two intentional surfaces: User and Administrator. Pre-production status — no legacy compatibility constraints.

---

## Role-Capability Matrix

**User:**
- Chat and workspace usage
- Personal settings and preferences
- Personal memory usage

**Administrator:**
- Everything User can do
- Diagnostics and observability exploration
- Non-destructive tuning and analysis workflows
- Global maintenance and destructive operations
- Full setup and governance surface

Default for unknown role: User-safe capabilities only.

---

## Information Architecture

**User surface (all authenticated users):**
- `/` — Chat + workspace
- `/memories` — Personal memory management
- `/settings` — Language, sound, tool toggles, RAG on/off + top_k, chat model preference

**Admin surface (administrator role only):**
- `/admin` — LLM diagnostics, vision/auxiliary model selection, RAG collection + threshold, re-ingest all documents, reset all databases (danger zone)
- `/metrics` — System analytics and observability (admin-only, middleware-gated)

**Navigation (Header.tsx):**
- Always visible: Memories, Settings
- Admin only: Setup (`/admin`), Metrics (`/metrics`) — rendered via `useRole()`

---

## Access-Control Contract

**Role encoding in JWT:**
Add `role: "user" | "administrator"` to `SessionUser` in `lib/auth/session.ts`. Encode it in the JWT at login by reading `AUTH_USER_ROLE` env var (default `"user"`).

**Middleware route guard (`proxy.ts`):**

```ts
const ADMIN_ROUTES = new Set(["/admin", "/metrics"]);
```

After session validation, if `pathname` matches `ADMIN_ROUTES` and `session.role !== "administrator"`, redirect to `/`. This is the only place for hard enforcement — client-side guards are UX-only.

**When `AUTH_ENABLED=false` (local dev):**
`RoleContext` reads `NEXT_PUBLIC_AUTH_USER_ROLE` env var (default `"user"`). Set it to `"administrator"` locally to access admin features without enabling full auth.

**Client-side RoleContext (`context/RoleContext.tsx`):**
React context that calls `GET /api/auth/session`, stores the role, and provides a `useRole()` hook. Wired into `app/layout.tsx` alongside existing providers. Used by `Header.tsx` and `AdminOnly.tsx`.

**Section guard (`components/AdminOnly.tsx`):**
Renders children only when `useRole()` returns `"administrator"`. Returns `null` otherwise. Never renders admin/destructive controls for non-admin roles.

---

## Control Split

### Personal Settings (`/settings`)

| Control | Note |
| --- | --- |
| Language selector | Top-level field — safe to save independently |
| Sound toggle | In `analytics_preferences.sound_enabled` |
| Tool toggles | Top-level field — safe to save independently |
| RAG enabled + top_k | User retrieval tuning |
| Chat model selector | Personal preference |

### Admin / Setup (`/admin`)

| Control | Section |
| --- | --- |
| RAG collection name + score threshold | Operational tuning |
| Re-ingest all documents | Operational (confirm dialog) |
| LLM capability diagnostics table + copy/export | Diagnostics |
| Vision + auxiliary model selectors | System model roles |
| Reset all databases | Danger zone (confirm dialog + checkbox) |

### Analytics / Observability (`/metrics`)

Full metrics page content unchanged — realtime and trends tabs, CSV export, chart preferences. Admin-only access.

---

## Critical Implementation Note: Read-Modify-Write on Every Save

The `POST /api/settings/user/{user_id}` backend endpoint **replaces** dict fields wholesale — it does not deep-merge. Confirmed in `backend/routers/settings.py`: `_resolved_value` reads the Pydantic model's parsed value, which defaults to `{}` for any missing dict field.

Consequence: if the User settings page sends only `{ preferred_models: { chat: "x" } }`, it wipes admin-set `vision` and `auxiliary` keys. Same problem for any partial `rag_config` or `analytics_preferences` save.

**Fix:** Both `SettingsPanel` and `AdminPanel` must:
1. Hydrate the **full** `UserSettings` object from the server on mount (via `useSettings` hook, already fetches on mount).
2. Display and allow editing of only their allowed subset of controls.
3. Send the **complete merged settings object** on save — all fields, not just the modified ones.

This preserves the other page's values without any backend changes.

The same applies to `analytics_preferences`: the User settings page manages `sound_enabled`, the Metrics page manages chart preferences. Both must read the full object on load and only modify their own key before saving.

---

## UX Safeguards

- All destructive controls are in a dedicated danger zone section on `/admin`.
- Destructive actions use `ConfirmDialog` with `variant="danger"` and `requireChecked` (component already exists — reuse it).
- Clear consequence summaries shown before action execution.

---

## Delivery Sequence

1. **Role in JWT** — `lib/auth/session.ts`, `app/api/auth/login/route.ts`, `app/api/auth/session/route.ts`
2. **Middleware role guard** — `proxy.ts` (add `ADMIN_ROUTES`, role check)
3. **RoleContext + useRole()** — new `context/RoleContext.tsx`, wired in `app/layout.tsx`
4. **Admin page shell** — new `app/admin/page.tsx` (empty, confirm gate works)
5. **Decompose SettingsPanel** — new `components/AdminPanel.tsx` with admin controls extracted; `SettingsPanel` trimmed to user controls only
6. **Update Header nav** — add conditional `/admin` + `/metrics` links gated on `useRole()`
7. **Component-level guards** — `components/AdminOnly.tsx`
8. **UX guardrails** — danger zone layout, confirm dialogs wired
9. **i18n** — add translation keys in `i18n/messages.ts` for new headings and labels
10. **Tests** — update `components/__tests__/SettingsPanel.test.tsx` for decomposed components
11. **End-to-end role journey validation**

---

## Files to Create or Modify

| File | Action | What changes |
| --- | --- | --- |
| `lib/auth/session.ts` | Modify | Add `role` to `SessionUser`, JWT encode/decode |
| `app/api/auth/login/route.ts` | Modify | Read `AUTH_USER_ROLE` env var, pass role to `createSessionToken` |
| `app/api/auth/session/route.ts` | Modify | Return `role` in session response body |
| `proxy.ts` | Modify | Add `ADMIN_ROUTES` set + role check after session validation |
| `context/RoleContext.tsx` | Create | `RoleContext`, `RoleProvider`, `useRole()` hook |
| `app/layout.tsx` | Modify | Wrap with `RoleProvider` |
| `app/admin/page.tsx` | Create | Admin/Setup page shell |
| `components/AdminPanel.tsx` | Create | Admin controls extracted from `SettingsPanel` |
| `components/SettingsPanel.tsx` | Modify | Remove admin controls, keep user controls only |
| `components/AdminOnly.tsx` | Create | Guard component: renders children only for admin role |
| `components/Header.tsx` | Modify | Conditional admin + metrics nav links via `useRole()` |
| `app/profile/page.tsx` | Delete | Unused redirect — no longer needed |
| `i18n/messages.ts` | Modify | Add translation keys for admin page headings and labels |
| `components/__tests__/SettingsPanel.test.tsx` | Modify | Update for decomposed components |
| `.env.example` / docker env | Document | Add `NEXT_PUBLIC_AUTH_USER_ROLE` (default `user`) |

---

## Verification

1. **Auth enabled, user role:** `AUTH_ENABLED=true`, `AUTH_USER_ROLE=user` → login → `/admin` and `/metrics` redirect to `/`, admin nav links absent, destructive controls not rendered anywhere.
2. **Auth enabled, admin role:** `AUTH_USER_ROLE=administrator` → login → `/admin` and `/metrics` load, all admin controls visible, user settings page has no admin controls.
3. **Auth disabled dev mode:** `AUTH_ENABLED=false`, `NEXT_PUBLIC_AUTH_USER_ROLE=administrator` → admin nav links visible, `/admin` and `/metrics` load. Without the env var, defaults to user role.
4. **Save scope test:** As admin, set vision model to "llava". Save from user settings page. Re-check admin settings → vision model must still be "llava" (proves read-modify-write pattern works).
5. **ConfirmDialog wiring:** `variant="danger"` + `requireChecked` reused in `AdminPanel` for database reset.
6. `npm run lint` and `npm run test` after each phase.

---

## Relevant Files (Reference)

- `frontend/src/components/SettingsPanel.tsx` — source of mixed controls to decompose
- `frontend/src/components/Header.tsx` — role-aware navigation composition
- `frontend/src/lib/auth/session.ts` — JWT session management, role claim source
- `frontend/src/proxy.ts` — middleware, route-level role guard insertion point
- `frontend/src/app/settings/page.tsx` — target Personal Settings shell
- `frontend/src/app/metrics/page.tsx` — observability content, becomes admin-only
- `frontend/src/app/memories/page.tsx` — user-safe surface, unchanged
- `frontend/src/lib/api.ts` — API action mapping
- `backend/routers/settings.py` — settings endpoint (full-replace behavior, no deep merge)

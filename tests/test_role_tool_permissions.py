"""Tests for admin-controlled, role-based tool permissions.

Covers the tool catalog grouping, the admin role-tools endpoints, the
``allowed_tools`` surfaced by ``/settings/me``, and the server-side enforcement
in ``node_load_tools``. The real DB-backed ``RoleToolPermissionStore`` is exercised
via integration tests elsewhere; here we use a lightweight fake store.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

import universal_agentic_framework.orchestration.graph_builder as gb
from backend.routers.settings import router, _load_tool_catalog, _catalog_tool_ids


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeRoleToolStore:
    def __init__(self, initial: Optional[Dict[str, List[str]]] = None) -> None:
        self._data: Dict[str, List[str]] = {k: list(v) for k, v in (initial or {}).items()}

    def get_allowed_tools(self, role_name: str) -> Optional[List[str]]:
        if role_name not in self._data:
            return None
        return list(self._data[role_name])

    def get_all_role_permissions(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._data.items()}

    def set_allowed_tools(self, role_name: str, tools: List[str]) -> List[str]:
        self._data[role_name] = list(tools)
        return list(tools)


class _FakeSettingsStore:
    def __init__(self) -> None:
        self.record: Optional[Dict] = None

    def get_user_settings(self, user_id: str):
        _ = user_id
        return dict(self.record) if self.record else None

    def upsert_user_settings(self, **kwargs):
        self.record = {
            "user_id": kwargs["user_id"],
            "tool_toggles": kwargs["tool_toggles"],
            "rag_config": kwargs["rag_config"],
            "analytics_preferences": kwargs.get("analytics_preferences", {}),
            "preferred_model": kwargs.get("preferred_model"),
            "preferred_models": kwargs.get("preferred_models", {}),
            "theme": kwargs.get("theme", "auto"),
            "language": kwargs.get("language", "en"),
            "updated_at": None,
        }
        return dict(self.record)


def _make_client(
    monkeypatch,
    *,
    role: str = "administrator",
    role_store: Optional[_FakeRoleToolStore] = None,
    settings_store: Optional[_FakeSettingsStore] = None,
) -> TestClient:
    monkeypatch.setenv("PROFILE_ID", "starter")
    monkeypatch.delenv("AUTH_ENABLED", raising=False)  # dev bypass → role from env
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("NEXT_PUBLIC_AUTH_USER_ROLE", role)
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    app = FastAPI()
    app.include_router(router)
    app.state.role_tool_store = role_store if role_store is not None else _FakeRoleToolStore()
    app.state.settings_store = settings_store if settings_store is not None else _FakeSettingsStore()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Catalog grouping
# --------------------------------------------------------------------------- #
def test_catalog_groups_resolve_from_manifest_category(monkeypatch):
    monkeypatch.setenv("PROFILE_ID", "starter")
    catalog = _load_tool_catalog()
    by_id = {item["id"]: item for item in catalog}

    assert "mcp_stub" not in by_id  # deleted tool
    assert by_id["datetime_tool"]["group"] == "auxiliary"
    assert by_id["calculator_tool"]["group"] == "auxiliary"
    assert by_id["map_tool"]["group"] == "auxiliary"
    assert by_id["analyze_image_tool"]["group"] == "vision"
    assert by_id["ocr_tool"]["group"] == "vision"
    assert by_id["web_search_mcp"]["group"] == "text"
    assert by_id["extract_webpage_mcp"]["group"] == "text"
    assert by_id["file_ops_tool"]["group"] == "text"
    assert by_id["csv_analyze_tool"]["group"] == "text"
    assert all(item["group"] in {"text", "vision", "auxiliary"} for item in catalog)


# --------------------------------------------------------------------------- #
# Admin endpoints
# --------------------------------------------------------------------------- #
def test_admin_get_role_tools_returns_catalog_and_roles(monkeypatch):
    store = _FakeRoleToolStore({"user": ["datetime_tool"], "researcher": []})
    client = _make_client(monkeypatch, role="administrator", role_store=store)
    body = client.get("/api/admin/role-tools").json()

    assert body["roles"]["user"] == ["datetime_tool"]
    assert body["roles"]["researcher"] == []
    assert any(t["id"] == "datetime_tool" for t in body["tools"])
    assert all("group" in t and "label" in t for t in body["tools"])


def test_admin_get_role_tools_unconfigured_role_is_empty(monkeypatch):
    client = _make_client(monkeypatch, role="administrator", role_store=_FakeRoleToolStore())
    body = client.get("/api/admin/role-tools").json()
    assert body["roles"]["user"] == []
    assert body["roles"]["researcher"] == []


def test_role_tools_endpoints_require_admin(monkeypatch):
    client = _make_client(monkeypatch, role="researcher")
    assert client.get("/api/admin/role-tools").status_code == 403
    assert client.put("/api/admin/role-tools/user", json={"allowed_tools": []}).status_code == 403


def test_update_role_tools_rejects_administrator(monkeypatch):
    client = _make_client(monkeypatch, role="administrator")
    resp = client.put("/api/admin/role-tools/administrator", json={"allowed_tools": []})
    assert resp.status_code == 400


def test_update_role_tools_rejects_unknown_ids(monkeypatch):
    client = _make_client(monkeypatch, role="administrator")
    resp = client.put(
        "/api/admin/role-tools/user",
        json={"allowed_tools": ["datetime_tool", "does_not_exist"]},
    )
    assert resp.status_code == 400


def test_update_role_tools_stores_in_catalog_order(monkeypatch):
    store = _FakeRoleToolStore()
    client = _make_client(monkeypatch, role="administrator", role_store=store)
    # Submit out of order — stored result should follow catalog order.
    resp = client.put(
        "/api/admin/role-tools/user",
        json={"allowed_tools": ["calculator_tool", "datetime_tool"]},
    )
    assert resp.status_code == 200
    expected = [t for t in _catalog_tool_ids() if t in {"calculator_tool", "datetime_tool"}]
    assert store.get_allowed_tools("user") == expected
    assert set(resp.json()["roles"]["user"]) == {"calculator_tool", "datetime_tool"}


def test_update_role_tools_full_list_is_stored_explicitly(monkeypatch):
    store = _FakeRoleToolStore()
    client = _make_client(monkeypatch, role="administrator", role_store=store)
    all_ids = _catalog_tool_ids()
    resp = client.put("/api/admin/role-tools/user", json={"allowed_tools": all_ids})
    assert resp.status_code == 200
    # "all checked" is stored verbatim (never cleared) so it stays distinct from block-all.
    assert store.get_allowed_tools("user") == all_ids


# --------------------------------------------------------------------------- #
# /settings/me allowed_tools
# --------------------------------------------------------------------------- #
def test_settings_me_admin_gets_all_tools(monkeypatch):
    store = _FakeRoleToolStore({"user": ["datetime_tool"]})
    client = _make_client(monkeypatch, role="administrator", role_store=store)
    body = client.get("/api/settings/me").json()
    assert set(body["allowed_tools"]) == set(_catalog_tool_ids())


def test_settings_me_role_uses_store(monkeypatch):
    store = _FakeRoleToolStore({"user": ["datetime_tool", "calculator_tool"]})
    client = _make_client(monkeypatch, role="user", role_store=store)
    body = client.get("/api/settings/me").json()
    assert set(body["allowed_tools"]) == {"datetime_tool", "calculator_tool"}


def test_settings_me_missing_row_blocks_all(monkeypatch):
    client = _make_client(monkeypatch, role="user", role_store=_FakeRoleToolStore())
    body = client.get("/api/settings/me").json()
    assert body["allowed_tools"] == []


def test_allowed_tools_not_writable_via_post(monkeypatch):
    store = _FakeRoleToolStore({"user": ["datetime_tool"]})
    client = _make_client(monkeypatch, role="user", role_store=store)
    resp = client.post(
        "/api/settings/me",
        json={"tool_toggles": {}, "allowed_tools": ["calculator_tool", "web_search_mcp"]},
    )
    assert resp.status_code == 200
    # The injected allowed_tools is ignored; the server recomputes from the role store.
    assert set(resp.json()["allowed_tools"]) == {"datetime_tool"}


# --------------------------------------------------------------------------- #
# node_load_tools enforcement
# --------------------------------------------------------------------------- #
def _fake_discovery(*_args, **_kwargs):
    return [SimpleNamespace(name=n) for n in ("datetime_tool", "calculator_tool", "map_tool")]


def _loaded_names(state) -> set:
    return {t.name for t in state.get("loaded_tools", [])}


def _base_state(**overrides) -> Dict:
    state = {"messages": [{"role": "user", "content": "hi"}], "user_id": "u", "language": "en"}
    state.update(overrides)
    return state


def test_node_load_tools_none_allows_all(monkeypatch):
    monkeypatch.setenv("PROFILE_ID", "starter")
    monkeypatch.setattr(gb, "_discover_tools_cached", _fake_discovery)
    result = gb.node_load_tools(_base_state(allowed_tools=None))
    assert _loaded_names(result) == {"datetime_tool", "calculator_tool", "map_tool"}


def test_node_load_tools_empty_blocks_all(monkeypatch):
    monkeypatch.setenv("PROFILE_ID", "starter")
    monkeypatch.setattr(gb, "_discover_tools_cached", _fake_discovery)
    result = gb.node_load_tools(_base_state(allowed_tools=[]))
    assert _loaded_names(result) == set()


def test_node_load_tools_partial_intersects(monkeypatch):
    monkeypatch.setenv("PROFILE_ID", "starter")
    monkeypatch.setattr(gb, "_discover_tools_cached", _fake_discovery)
    result = gb.node_load_tools(_base_state(allowed_tools=["datetime_tool"]))
    assert _loaded_names(result) == {"datetime_tool"}


def test_node_load_tools_applies_toggles_on_top_of_role_gate(monkeypatch):
    monkeypatch.setenv("PROFILE_ID", "starter")
    monkeypatch.setattr(gb, "_discover_tools_cached", _fake_discovery)
    result = gb.node_load_tools(
        _base_state(
            allowed_tools=["datetime_tool", "calculator_tool"],
            user_settings={"tool_toggles": {"calculator_tool": False}},
        )
    )
    assert _loaded_names(result) == {"datetime_tool"}

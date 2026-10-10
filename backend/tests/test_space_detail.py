"""Tests for GET /api/spaces/{space_id} (space detail).

The Genie API payload carries no UI link, so the endpoint adds ``space_url``
(like the list endpoint) — otherwise a page opened by URL / reloaded loses
every Genie link (Agent header, Benchmark tab "Edit in Genie").
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import spaces

SPACE_ID = "01f1b05f348113e6a637419c8abf4020"


class _FakeApiClient:
    def __init__(self, payload: dict):
        self._payload = payload

    def do(self, method: str, path: str, **_kwargs):
        assert method == "GET"
        assert path == f"/api/2.0/genie/spaces/{SPACE_ID}"
        return dict(self._payload)


def _fake_client(host: str | None, payload: dict | None = None):
    return SimpleNamespace(
        config=SimpleNamespace(host=host),
        api_client=_FakeApiClient(payload or {"space_id": SPACE_ID, "title": "Nuvemshop"}),
    )


@pytest.fixture
def client(monkeypatch):
    async def _score(_space_id):
        return None

    async def _starred(_space_id):
        return False

    monkeypatch.setattr(spaces, "get_latest_score", _score)
    monkeypatch.setattr(spaces, "is_space_starred", _starred)
    app = FastAPI()
    app.include_router(spaces.router)
    return TestClient(app)


def test_detail_adds_space_url_from_workspace_host(client, monkeypatch) -> None:
    monkeypatch.setattr(spaces, "get_workspace_client", lambda: _fake_client("https://ws.example.com/"))
    resp = client.get(f"/api/spaces/{SPACE_ID}")
    assert resp.status_code == 200
    assert resp.json()["space"]["space_url"] == f"https://ws.example.com/genie/rooms/{SPACE_ID}"


def test_detail_keeps_existing_space_url(client, monkeypatch) -> None:
    payload = {"space_id": SPACE_ID, "space_url": "https://custom/link"}
    monkeypatch.setattr(spaces, "get_workspace_client", lambda: _fake_client("https://ws.example.com", payload))
    resp = client.get(f"/api/spaces/{SPACE_ID}")
    assert resp.json()["space"]["space_url"] == "https://custom/link"


def test_detail_without_host_leaves_space_url_unset(client, monkeypatch) -> None:
    monkeypatch.setattr(spaces, "get_workspace_client", lambda: _fake_client(None))
    resp = client.get(f"/api/spaces/{SPACE_ID}")
    assert resp.status_code == 200
    assert "space_url" not in resp.json()["space"]

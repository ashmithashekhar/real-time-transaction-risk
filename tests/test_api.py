"""API tests using FastAPI's TestClient against a temp SQLite database."""

import pytest
from fastapi.testclient import TestClient

from app import models
from app.main import app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("api") / "api.db"
    models.init_db(f"sqlite:///{db_file}")
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_merchants_seeded_on_startup(client):
    resp = client.get("/merchants")
    assert resp.status_code == 200
    merchants = resp.json()
    assert len(merchants) == 24
    assert {"id", "name", "category", "region"} <= set(merchants[0])


def test_analyze_and_fetch_report(client):
    merchant_id = client.get("/merchants").json()[0]["id"]

    resp = client.post(f"/merchants/{merchant_id}/analyze")
    assert resp.status_code == 200
    report = resp.json()
    assert report["merchant_id"] == merchant_id
    assert report["risk_level"] in {"low", "moderate", "high", "critical"}
    assert report["summary"]
    assert "profile" in report["findings"]

    fetched = client.get(f"/reports/{report['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == report["id"]

    listed = client.get(f"/merchants/{merchant_id}/reports")
    assert listed.status_code == 200
    assert any(r["id"] == report["id"] for r in listed.json())


def test_not_found_paths(client):
    assert client.get("/merchants/999999").status_code == 404
    assert client.post("/merchants/999999/analyze").status_code == 404
    assert client.get("/reports/999999").status_code == 404

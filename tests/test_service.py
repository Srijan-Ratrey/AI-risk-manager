"""Service tests, including the degraded path.

The fallback is the part most likely to be claimed and never exercised, so
it gets a real test that removes the model and asserts the service still
returns a decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.service as service  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "AUDIT_DB", tmp_path / "audit.db")
    service.engine = None
    return TestClient(service.app)


@pytest.fixture
def degraded_client(tmp_path, monkeypatch):
    """Same service, but the model file is unreachable."""
    monkeypatch.setattr(service, "AUDIT_DB", tmp_path / "audit.db")
    monkeypatch.setattr(service, "MODEL_PATH", tmp_path / "does-not-exist.pkl")
    service.engine = None
    return TestClient(service.app)


def test_health_reports_model_loaded(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_score_returns_a_bounded_decision(client):
    body = client.post("/v1/score", json={
        "transaction_id": "txn_test_001",
        "amount_inr": 12_000.0,
        "features": {"TransactionAmt": 136.0, "ProductCD": "W", "C1": 1, "C12": 0, "C13": 1},
    }).json()

    assert body["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["reason_codes"]
    assert body["degraded"] is False
    assert body["model_version"] == service.MODEL_VERSION
    assert body["cost_model_version"]
    # Never leak the internal feature set to the merchant-facing string.
    assert "C12" not in body["merchant_message"]


def test_high_count_signal_scores_above_a_clean_transaction(client):
    def score_of(features):
        return client.post("/v1/score", json={
            "amount_inr": 12_000.0, "features": features
        }).json()["score"]

    clean = score_of({"TransactionAmt": 136.0, "ProductCD": "W", "C1": 1, "C12": 0, "C13": 1})
    bursty = score_of({"TransactionAmt": 136.0, "ProductCD": "W", "C1": 40, "C12": 30, "C13": 50})
    assert bursty > clean


def test_decision_is_written_to_the_audit_log(client):
    client.post("/v1/score", json={
        "transaction_id": "txn_audit_me", "amount_inr": 5_000.0, "features": {"C12": 0},
    })
    rows = client.get("/v1/audit/txn_audit_me").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert row["model_version"] and row["cost_model_version"]
    assert row["latency_ms"] > 0
    assert row["overturned_to"] is None


def test_appeal_is_recorded_not_deleted(client):
    client.post("/v1/score", json={
        "transaction_id": "txn_appeal", "amount_inr": 90_000.0,
        "features": {"C1": 40, "C12": 30, "C13": 50},
    })
    client.post("/v1/appeal/txn_appeal", params={"overturn_to": "APPROVE"})
    rows = client.get("/v1/audit/txn_appeal").json()
    assert len(rows) == 1, "appeal must not delete the original decision"
    assert rows[0]["overturned_to"] == "APPROVE"


# -- degraded mode -------------------------------------------------------


def test_service_starts_without_a_model(degraded_client):
    body = degraded_client.get("/v1/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_degraded_service_still_decides_and_does_not_fail_open(degraded_client):
    """The rule must actually fire, not just return APPROVE for everything."""
    clean = degraded_client.post("/v1/score", json={
        "amount_inr": 5_000.0, "features": {"C12": 0},
    }).json()
    bursty = degraded_client.post("/v1/score", json={
        "amount_inr": 5_000.0, "features": {"C12": 30},
    }).json()

    assert clean["degraded"] is True and bursty["degraded"] is True
    assert clean["decision"] == "APPROVE"
    assert bursty["decision"] == "BLOCK"
    assert "DEGRADED" in bursty["reason_codes"][0]
    assert bursty["model_version"] == "fallback-rule-1.0.0"


def test_degraded_decisions_are_audited_too(degraded_client):
    degraded_client.post("/v1/score", json={
        "transaction_id": "txn_degraded", "amount_inr": 5_000.0, "features": {"C12": 30},
    })
    rows = degraded_client.get("/v1/audit/txn_degraded").json()
    assert rows[0]["degraded"] == 1

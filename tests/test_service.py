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
    body = client.get("/v1/audit/txn_audit_me").json()
    assert len(body["decisions"]) == 1
    assert body["appeals"] == []
    row = body["decisions"][0]
    assert row["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert row["model_version"] and row["cost_model_version"]
    assert row["latency_ms"] > 0


def test_appeal_appends_and_leaves_the_decision_untouched(client):
    """The audit log is append-only: an overturn must not rewrite history.

    Retraining needs the pair -- what the model said and what the human said
    -- so overwriting the decision destroys the label being collected.
    """
    client.post("/v1/score", json={
        "transaction_id": "txn_appeal", "amount_inr": 90_000.0,
        "features": {"C1": 40, "C12": 30, "C13": 50},
    })
    before = client.get("/v1/audit/txn_appeal").json()["decisions"][0]

    response = client.post("/v1/appeal/txn_appeal", params={"overturn_to": "APPROVE"})
    assert response.status_code == 201

    body = client.get("/v1/audit/txn_appeal").json()
    assert body["decisions"] == [before], "the original decision row must be untouched"
    assert len(body["appeals"]) == 1
    assert body["appeals"][0]["overturn_to"] == "APPROVE"


def test_appeal_rejects_an_invalid_decision(client):
    client.post("/v1/score", json={
        "transaction_id": "txn_bad_appeal", "amount_inr": 500.0, "features": {"C12": 0},
    })
    response = client.post("/v1/appeal/txn_bad_appeal", params={"overturn_to": "BANANA"})
    assert response.status_code == 422
    assert client.get("/v1/audit/txn_bad_appeal").json()["appeals"] == []


def test_appeal_on_an_unknown_transaction_is_404(client):
    response = client.post("/v1/appeal/txn_never_scored", params={"overturn_to": "APPROVE"})
    assert response.status_code == 404


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


def test_degraded_mode_escalates_when_its_own_input_is_missing(degraded_client):
    """The fallback must not treat an absent C12 as zero.

    Coercing missing to 0.0 scores 0.0 and APPROVES -- failing open on
    exactly the requests we know least about, while reporting a confident
    decision. Nothing in the request schema requires C12, so this is
    reachable with an ordinary payload.
    """
    for features in ({"TransactionAmt": 500.0}, {"C12": None}, {"C12": "not-a-number"}):
        body = degraded_client.post("/v1/score", json={
            "amount_inr": 50_000.0, "features": features,
        }).json()
        assert body["decision"] == "REVIEW", f"failed open on {features}"
        assert "escalated to review" in body["reason_codes"][0]


def test_degraded_audit_records_the_rule_threshold_not_the_band(degraded_client):
    """A replayed decision must show the cut that actually ran."""
    body = degraded_client.post("/v1/score", json={
        "transaction_id": "txn_degraded", "amount_inr": 5_000.0, "features": {"C12": 30},
    }).json()
    assert body["threshold_used"] == service.FALLBACK_THRESHOLD

    rows = degraded_client.get("/v1/audit/txn_degraded").json()["decisions"]
    assert rows[0]["degraded"] == 1
    assert rows[0]["threshold_used"] == service.FALLBACK_THRESHOLD


def test_missing_thresholds_degrade_rather_than_serving_a_guessed_band(tmp_path, monkeypatch):
    """A hardcoded stand-in band would serve a different policy while
    reporting itself healthy. Absent thresholds must degrade instead."""
    monkeypatch.setattr(service, "AUDIT_DB", tmp_path / "audit.db")
    monkeypatch.setattr(service, "THRESHOLD_PATH", tmp_path / "missing.json")
    service.engine = None
    client = TestClient(service.app)

    assert client.get("/v1/health").json()["status"] == "degraded"
    body = client.post("/v1/score", json={"amount_inr": 500.0, "features": {"C12": 0}}).json()
    assert body["degraded"] is True

"""Replay real test-window transactions through the service.

Doubles as the latency benchmark and the video demo. Payments are decided in
the checkout path, so the honest thing to report is a measured p95 under
sequential load, not a hand-wave that "trees are fast".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import src.service as service
from src.costs import CostModel
from src.data import feature_columns, load_raw, make_splits

N_REQUESTS = 1000


def main() -> None:
    costs = CostModel.load()
    df = load_raw()
    splits = make_splits(df)
    test = df.iloc[splits.test]
    names = feature_columns(df)

    rng = np.random.default_rng(42)
    # Oversample fraud so the demo shows all three decisions; latency is
    # measured over the same sample and is not sensitive to the mix.
    fraud_idx = np.flatnonzero(test["isFraud"].to_numpy() == 1)
    legit_idx = np.flatnonzero(test["isFraud"].to_numpy() == 0)
    sample = np.concatenate([
        rng.choice(fraud_idx, size=N_REQUESTS // 4, replace=False),
        rng.choice(legit_idx, size=N_REQUESTS - N_REQUESTS // 4, replace=False),
    ])
    rng.shuffle(sample)

    client = TestClient(service.app)
    health = client.get("/v1/health").json()
    print(f"service: {health['status']}  model={health['model_version']}  "
          f"costs={health['cost_model_version']}")

    latencies, decisions, records = [], [], []
    for position in sample:
        row = test.iloc[int(position)]
        features = {
            name: (None if pd.isna(row[name]) else
                   (row[name].item() if hasattr(row[name], "item") else row[name]))
            for name in names
        }
        payload = {
            "transaction_id": f"txn_{int(row['TransactionID'])}",
            "amount_inr": float(row["TransactionAmt"]) * costs.usd_to_inr,
            "features": features,
        }
        started = time.perf_counter()
        body = client.post("/v1/score", json=payload).json()
        latencies.append((time.perf_counter() - started) * 1000.0)

        decisions.append(body["decision"])
        records.append({
            "transaction_id": body["transaction_id"],
            "amount_inr": payload["amount_inr"],
            "is_fraud": int(row["isFraud"]),
            "score": body["score"],
            "decision": body["decision"],
            "reason_codes": " | ".join(body["reason_codes"]),
        })

    latencies = np.array(latencies)
    served = pd.DataFrame(records)

    print(f"\n=== latency over {len(latencies)} sequential requests ===")
    print(f"  mean {latencies.mean():.2f} ms")
    print(f"  p50  {np.percentile(latencies, 50):.2f} ms")
    print(f"  p95  {np.percentile(latencies, 95):.2f} ms")
    print(f"  p99  {np.percentile(latencies, 99):.2f} ms")
    print(f"  max  {latencies.max():.2f} ms")
    budget = "MET" if np.percentile(latencies, 95) < 100 else "MISSED"
    print(f"  p95 < 100 ms budget: {budget}")
    print("  (includes SHAP reason codes and the audit-log write)")

    print("\n=== decision mix ===")
    mix = served.groupby("decision").agg(
        n=("decision", "size"), fraud_rate=("is_fraud", "mean"),
        mean_score=("score", "mean"),
    )
    print(mix.to_string(float_format=lambda v: f"{v:,.4f}"))

    print("\n=== sample decisions ===")
    for decision in ("APPROVE", "REVIEW", "BLOCK"):
        subset = served[served["decision"] == decision]
        if subset.empty:
            continue
        row = subset.iloc[0]
        print(f"\n{decision}  txn {row['transaction_id']}  Rs {row['amount_inr']:,.0f}  "
              f"score {row['score']:.4f}  actually_fraud={bool(row['is_fraud'])}")
        for code in row["reason_codes"].split(" | "):
            print(f"    - {code}")

    Path("reports").mkdir(exist_ok=True)
    served.to_csv("reports/demo_decisions.csv", index=False)
    Path("reports/latency.json").write_text(json.dumps({
        "n_requests": len(latencies),
        "mean_ms": float(latencies.mean()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "max_ms": float(latencies.max()),
        "budget_ms": 100, "budget_met": bool(np.percentile(latencies, 95) < 100),
    }, indent=2))
    print("\nwritten to reports/demo_decisions.csv and reports/latency.json")


if __name__ == "__main__":
    main()

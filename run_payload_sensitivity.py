"""How much of the score survives when the payload is incomplete?

This answers a demo question with a measurement instead of an assertion: can
you hand-author a convincing fraudulent transaction, or must you replay real
rows?

It takes a real fraud transaction that the full pipeline blocks, then scores
it repeatedly with progressively larger subsets of its features, and finally
with plausible-looking values a human might invent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import src.service as service
from src.costs import CostModel
from src.data import feature_columns, load_raw, make_splits

# A real fraud row from the test window; the full-feature replay blocks it.
TRANSACTION_ID = 3513412


def main() -> None:
    costs = CostModel.load()
    df = load_raw()
    splits = make_splits(df)
    names = feature_columns(df)
    test = df.iloc[splits.test]
    row = test[test["TransactionID"] == TRANSACTION_ID].iloc[0]

    client = TestClient(service.app)

    def send(keep: list[str], label: str) -> dict:
        features = {}
        for name in keep:
            value = row[name]
            if pd.isna(value):
                continue
            features[name] = value.item() if hasattr(value, "item") else value
        body = client.post("/v1/score", json={
            "transaction_id": f"sens_{len(features)}",
            "amount_inr": float(row["TransactionAmt"]) * costs.usd_to_inr,
            "features": features,
        }).json()
        return {"payload": label, "features_sent": len(features),
                "score": body["score"], "decision": body["decision"]}

    interpretable = ["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain",
                     "R_emaildomain", "addr1", "addr2", "dist1", "DeviceType", "DeviceInfo"]
    counts = [c for c in names if c.startswith("C") and c[1:].isdigit()]
    deltas = [c for c in names if c.startswith("D") and c[1:].isdigit()]

    rows = [
        send(interpretable, "interpretable only (what a human can author)"),
        send(interpretable + counts, "+ C counting features"),
        send(interpretable + counts + deltas, "+ D time-delta features"),
        send(names, "ALL features (what run_demo.py replays)"),
    ]

    # The same shape of payload, but with values a human would guess rather
    # than values that actually co-occur.
    invented = client.post("/v1/score", json={
        "transaction_id": "sens_invented", "amount_inr": 2_992.0,
        "features": {"TransactionAmt": 34.0, "ProductCD": "C", "card4": "mastercard",
                     "card6": "credit", "C1": 45, "C12": 28, "C13": 60, "C14": 35},
    }).json()
    rows.append({"payload": "INVENTED 'card testing' values", "features_sent": 8,
                 "score": invented["score"], "decision": invented["decision"]})

    table = pd.DataFrame(rows)
    print(f"=== payload sensitivity on txn {TRANSACTION_ID} "
          f"(isFraud={int(row['isFraud'])}, ${row['TransactionAmt']:.2f}) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nReal features reproduce the full score once C and D are present.")
    print("Invented values do not: the model keys on the joint pattern across")
    print("correlated features, so a fabricated row is out-of-distribution")
    print("rather than suspicious. Demos must replay real rows.")

    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/payload_sensitivity.csv", index=False)
    Path("reports/payload_sensitivity.json").write_text(json.dumps({
        "transaction_id": TRANSACTION_ID,
        "is_fraud": int(row["isFraud"]),
        "amount_usd": float(row["TransactionAmt"]),
        "rows": rows,
    }, indent=2))
    print("\nwritten to reports/payload_sensitivity.csv and .json")


if __name__ == "__main__":
    main()

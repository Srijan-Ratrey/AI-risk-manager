"""Persist the training feature schema so serving cannot drift from training.

The service builds a one-row frame from an arbitrary JSON payload. Unless it
reproduces the EXACT dtypes seen at training -- categorical columns with the
exact same category sets -- LightGBM rejects the frame. Inferring dtypes
from whatever the caller happened to send is precisely the train/serve skew
that makes a model behave differently in production than in the notebook.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from src.data import feature_columns, load_raw, prepare_features

OUT = Path("models/schema.pkl")


def main() -> None:
    df = load_raw()
    features = prepare_features(df)
    names = feature_columns(df)

    schema = {
        "feature_names": names,
        "dtypes": {name: features[name].dtype for name in names},
        "categorical": [
            name for name in names if str(features[name].dtype) == "category"
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump(schema, fh)

    print(f"features: {len(names)}")
    print(f"categorical: {len(schema['categorical'])} -> {schema['categorical']}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()

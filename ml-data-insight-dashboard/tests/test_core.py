import numpy as np
import pandas as pd

from ml_insights import cluster_summary, detect_anomalies, feature_importance
from preprocess import clean, split_columns


def _df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2,
                         "grp": rng.choice(["a", "b"], n),
                         "y": 5 * x1 + rng.normal(0, 0.3, n)})


def test_clean_drops_duplicates_and_empty_cols():
    df = pd.DataFrame({" a ": [1, 1, 2], "b": [None] * 3})
    out = clean(df)
    assert list(out.columns) == ["a"] and len(out) == 2


def test_split_columns():
    num, cat = split_columns(_df())
    assert "x1" in num and "grp" in cat


def test_feature_importance_ranks_true_driver_first():
    imp, score, task = feature_importance(_df(), "y")
    assert task == "regression" and score > 0.8
    assert imp.iloc[0]["feature"] == "x1"


def test_feature_importance_needs_rows():
    try:
        feature_importance(_df(10), "y")
        assert False
    except ValueError:
        pass


def test_anomalies_finds_injected_outliers():
    df = _df()
    df.loc[:2, "x1"] = 50
    flagged = detect_anomalies(df, contamination=0.02)
    assert {0, 1, 2} & set(flagged.index)


def test_cluster_summary_shape():
    s = cluster_summary(_df(), k=3)
    assert len(s) == 3 and s["size"].sum() == 300

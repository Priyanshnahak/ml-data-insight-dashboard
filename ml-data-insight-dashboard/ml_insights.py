"""ML-based insights: feature importance, anomaly detection, customer/segment clustering."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import build_preprocessor, split_columns

SEED = 42


def is_classification(y: pd.Series) -> bool:
    return (not pd.api.types.is_numeric_dtype(y)) or y.nunique() <= 5


def feature_importance(df: pd.DataFrame, target: str):
    """Fit a random forest and rank original columns by permutation importance.

    Returns (importance_df, held_out_score, task). Score is accuracy or R^2.
    """
    data = df.dropna(subset=[target])
    if len(data) < 20:
        raise ValueError("Need at least 20 rows with a non-missing target.")
    X, y = data.drop(columns=[target]), data[target]
    num, cat = split_columns(X)
    if not (num or cat):
        raise ValueError("No usable feature columns.")
    task = "classification" if is_classification(y) else "regression"
    model_cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    pipe = Pipeline([("prep", build_preprocessor(num, cat)),
                     ("model", model_cls(n_estimators=200, random_state=SEED, n_jobs=-1))])
    cols = num + cat
    X_tr, X_te, y_tr, y_te = train_test_split(X[cols], y, test_size=0.25, random_state=SEED)
    pipe.fit(X_tr, y_tr)
    score = float(pipe.score(X_te, y_te))
    res = permutation_importance(pipe, X_te, y_te, n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": cols, "importance": res.importances_mean})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    return imp, score, task


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """Rows flagged as unusual by an Isolation Forest over the numeric columns."""
    num = df.select_dtypes("number")
    if num.shape[1] == 0 or len(num) < 20:
        return df.iloc[0:0]
    X = SimpleImputer(strategy="median").fit_transform(num)
    flags = IsolationForest(contamination=contamination, random_state=SEED).fit_predict(X) == -1
    return df[flags]


def cluster_summary(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """KMeans segments on numeric columns; returns per-cluster means and sizes."""
    num = df.select_dtypes("number")
    if num.shape[1] < 2 or len(num) < k * 5:
        raise ValueError("Need at least 2 numeric columns and a few rows per cluster.")
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(num))
    labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
    out = num.copy()
    out["cluster"] = labels
    summary = out.groupby("cluster").mean().round(2)
    summary["size"] = out.groupby("cluster").size()
    return summary

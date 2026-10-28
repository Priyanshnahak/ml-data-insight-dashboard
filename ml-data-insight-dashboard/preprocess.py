"""Cleaning and preprocessing pipeline shared by all analyses."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_CATEGORIES = 50


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names, drop empty columns and duplicate rows."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.dropna(axis=1, how="all").drop_duplicates()
    return out.reset_index(drop=True)


def split_columns(df: pd.DataFrame):
    """Return (numeric_cols, categorical_cols). High-cardinality text columns are skipped."""
    num = df.select_dtypes("number").columns.tolist()
    cat = [c for c in df.columns if c not in num and df[c].nunique() <= MAX_CATEGORIES]
    return num, cat


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])

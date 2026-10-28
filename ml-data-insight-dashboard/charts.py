"""Automatic visualizations (matplotlib/seaborn figures)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def histograms(df, max_cols=6):
    cols = df.select_dtypes("number").columns[:max_cols]
    if len(cols) == 0:
        return None
    n = len(cols)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(c)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def correlation_heatmap(df):
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout()
    return fig


def category_counts(df, column, top=10):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    df[column].value_counts().head(top).plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_title(f"Top {top} values: {column}")
    fig.tight_layout()
    return fig

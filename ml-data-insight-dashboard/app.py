import pandas as pd
import streamlit as st

import charts
from ml_insights import cluster_summary, detect_anomalies, feature_importance
from preprocess import clean, split_columns

st.set_page_config(page_title="ML Data Insight Dashboard", layout="wide")
st.title("ML Data Insight Dashboard")
st.caption("Upload an Excel/CSV file to get automatic cleaning, charts and ML insights.")

file = st.file_uploader("Upload data", type=["xlsx", "xls", "csv"])
if not file:
    st.info("No file yet? Try sample_data/sales_sample.csv from the repo.")
    st.stop()

raw = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
df = clean(raw)
st.write(f"Loaded **{len(raw):,}** rows; **{len(df):,}** after removing duplicates/empty columns.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Charts", "Feature importance", "Anomalies & segments"])

with tab1:
    st.dataframe(df.head(50), use_container_width=True)
    st.write("Missing values per column")
    st.dataframe(df.isna().sum().rename("missing").to_frame())
    st.write("Summary statistics")
    st.dataframe(df.describe(include="all").T)

with tab2:
    fig = charts.histograms(df)
    if fig is not None:
        st.pyplot(fig)
    fig = charts.correlation_heatmap(df)
    if fig is not None:
        st.pyplot(fig)
    _, cats = split_columns(df)
    if cats:
        col = st.selectbox("Categorical column", cats)
        st.pyplot(charts.category_counts(df, col))

with tab3:
    target = st.selectbox("Column to explain/predict", df.columns)
    if st.button("Run analysis"):
        try:
            imp, score, task = feature_importance(df, target)
            label = "accuracy" if task == "classification" else "R²"
            st.write(f"Random forest ({task}) held-out {label}: **{score:.3f}**")
            st.bar_chart(imp.set_index("feature")["importance"])
            st.dataframe(imp)
        except ValueError as e:
            st.warning(str(e))

with tab4:
    anomalies = detect_anomalies(df)
    st.write(f"Isolation Forest flagged **{len(anomalies)}** unusual rows.")
    st.dataframe(anomalies)
    k = st.slider("Number of segments (KMeans)", 2, 6, 3)
    try:
        st.dataframe(cluster_summary(df, k))
    except ValueError as e:
        st.warning(str(e))

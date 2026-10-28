# ML Data Insight Dashboard

Upload an Excel or CSV file and get automatic cleaning, charts and machine-learning insights in one dashboard.

## Features
- **Preprocessing pipeline** (`preprocess.py`): removes duplicates/empty columns; median/mode imputation, scaling and one-hot encoding via scikit-learn pipelines.
- **Automatic visualizations** (`charts.py`): histograms, correlation heatmap, category counts.
- **Feature importance** (`ml_insights.py`): random forest + permutation importance shows which columns drive a chosen target (classification or regression, auto-detected).
- **Anomaly detection**: Isolation Forest flags unusual rows.
- **Segmentation**: KMeans clusters with per-segment profiles.

## Run
```bash
pip install -r requirements.txt
python make_sample_data.py        # optional: creates sample_data/sales_sample.csv
streamlit run app.py
```
Docker: `docker build -t ml-dashboard . && docker run -p 8501:8501 ml-dashboard`

## Tests
`pytest -q`

## Tech
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

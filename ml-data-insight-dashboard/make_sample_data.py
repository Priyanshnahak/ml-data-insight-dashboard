"""Generate a small synthetic sales dataset for demos and tests."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600
region = rng.choice(["North", "South", "East", "West"], n)
category = rng.choice(["Electronics", "Clothing", "Grocery"], n)
price = rng.uniform(5, 200, n).round(2)
discount = rng.choice([0, 5, 10, 20], n)
ad_spend = rng.uniform(0, 500, n).round(1)
units = (20 + 0.08 * ad_spend - 0.15 * price - 0.4 * discount * -1 + rng.normal(0, 4, n)).clip(1).round()
revenue = (units * price * (1 - discount / 100)).round(2)
df = pd.DataFrame(dict(region=region, category=category, price=price, discount=discount,
                       ad_spend=ad_spend, units=units, revenue=revenue))
df.loc[rng.choice(n, 25, replace=False), "ad_spend"] = np.nan   # missing values
df.loc[rng.choice(n, 4, replace=False), "revenue"] *= 12         # anomalies
df.to_csv("sample_data/sales_sample.csv", index=False)
print("wrote sample_data/sales_sample.csv", df.shape)

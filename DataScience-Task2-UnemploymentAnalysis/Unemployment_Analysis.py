import pandas as pd
import numpy as np

# 1. Create Sample Unemployment Dataset
data = {
    'Region': ['North', 'South', 'East', 'West', 'Central'] * 4,
    'Date': pd.date_range(start='2020-01-01', periods=20, freq='ME'),
    'Estimated Unemployment Rate (%)': [8.5, 9.1, 7.2, 6.8, 10.1, 15.2, 18.4, 21.0, 19.5, 12.3, 11.1, 9.8, 8.9, 7.5, 8.1, 6.9, 7.2, 6.5, 7.0, 6.2],
    'Estimated Employed': [100000, 120000, 95000, 110000, 85000] * 4,
    'Estimated Labour Participation Rate (%)': [40.2, 42.1, 39.5, 41.0, 38.8] * 4
}

df = pd.DataFrame(data)

# 2. Data Overview
print("Dataset Head:")
print(df.head())

# 3. Summary Statistics
print("\nData Summary:")
print(df.describe())

# 4. Region-wise Mean Unemployment Rate
region_unemployment = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
print("\nAverage Unemployment Rate by Region:")
print(region_unemployment)

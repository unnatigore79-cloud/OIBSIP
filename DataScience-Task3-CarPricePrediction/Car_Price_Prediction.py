import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset Creation
data = {
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'Present_Price': [5.59, 9.54, 9.85, 10.0, 6.87, 12.5, 8.0, 11.2],
    'Kms_Driven': [27000, 43000, 6900, 5200, 42450, 15000, 20000, 10000],
    'Fuel_Type': [0, 1, 0, 0, 1, 0, 1, 0],
    'Selling_Price': [3.35, 4.75, 7.25, 2.85, 4.60, 8.75, 6.00, 7.80]
}

df = pd.DataFrame(data)

# 2. Features and Target
X = df[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type']]
y = df['Selling_Price']

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

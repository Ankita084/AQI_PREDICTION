# ==========================================
# AIR QUALITY INDEX (AQI) PREDICTION PROJECT
# Complete End-to-End ML Code
# ==========================================

# STEP 1: Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# STEP 2: Load Dataset
# ==========================================

# Replace with your downloaded Kaggle file name

df = pd.read_csv("city_day.csv")

# ==========================================
# STEP 3: Show Data
# ==========================================

print(df.sample(10))

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# ==========================================
# STEP 4: Check Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# STEP 5: Remove Missing Values
# ==========================================

df = df.dropna()

# ==========================================
# STEP 6: EDA (Exploratory Data Analysis)
# ==========================================

# AQI Distribution

plt.figure(figsize=(8,5))
sns.histplot(df['AQI'], kde=True)
plt.title("AQI Distribution")
plt.show()

# ==========================================
# Correlation Heatmap
# ==========================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# Feature Importance Visual
# ==========================================

pollution_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']

for col in pollution_cols:
    if col not in df.columns:
        print(f"{col} column NOT found")

# ==========================================
# STEP 7: Feature Selection
# ==========================================

X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']]

y = df['AQI']

# ==========================================
# STEP 8: Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# STEP 9: Train Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# STEP 10: Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# STEP 11: Evaluation
# ==========================================

print("\nModel Performance")

print("MAE:", mean_absolute_error(y_test, y_pred))

print("MSE:", mean_squared_error(y_test, y_pred))

print("R2 Score:", r2_score(y_test, y_pred))

# ==========================================
# STEP 12: Feature Importance
# ==========================================

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8,5))

sns.barplot(
    x=importance,
    y=features
)

plt.title("Feature Importance")
plt.show()

# ==========================================
# STEP 13: Actual vs Predicted
# ==========================================

comparison = pd.DataFrame({
    'Actual AQI': y_test.values,
    'Predicted AQI': y_pred
})

print("\nActual vs Predicted")
print(comparison.head(60))

# ==========================================
# STEP 14: Predict New AQI
# ==========================================

# Example:
# PM2.5, PM10, NO2, SO2, CO

sample_data = [[80, 120, 40, 20, 1.2]]

prediction = model.predict(sample_data)

print("\nPredicted AQI:")
print(prediction)

# ==========================================
# STEP 15: Save Model
# ==========================================

import joblib

joblib.dump(model, "aqi_model.pkl")

print("\nModel Saved Successfully")

# ==========================================
# END PROJECT
# ==========================================
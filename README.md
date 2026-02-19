# FUTURE_ML_01


# ==========================================================
# SALES FORECASTING USING EXCEL FILE
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1️⃣ LOAD EXCEL FILE  (FIXED PATH)
file_path = r"C:\Users\UJWALA MANDLA\OneDrive\Desktop\future interns\sales.xlsx"
df = pd.read_excel(file_path)

print("Excel File Loaded Successfully")
print(df.head())

# 2️⃣ CLEAN DATA
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')
df = df.dropna()

# 3️⃣ FEATURE ENGINEERING
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek

# Lag Features
df['lag_1'] = df['Sales'].shift(1)
df['lag_7'] = df['Sales'].shift(7)

df = df.dropna()

# 4️⃣ SPLIT DATA (Time-based split)
X = df[['year','month','day','day_of_week','lag_1','lag_7']]
y = df['Sales']

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# 5️⃣ TRAIN MODEL
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6️⃣ PREDICT
predictions = model.predict(X_test)

# 7️⃣ EVALUATE
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nModel Performance:")
print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))

# 8️⃣ VISUALIZE
plt.figure(figsize=(12,6))
plt.plot(df['Date'].iloc[split:], y_test.values)
plt.plot(df['Date'].iloc[split:], predictions)
plt.title("Sales Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Actual Sales", "Predicted Sales"])
plt.show()

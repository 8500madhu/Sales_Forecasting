import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/store_sales.csv")

print("\n========== DATASET ==========")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())


# =========================================================
# 2. DATE PREPROCESSING
# =========================================================

df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date")

print("\nDate preprocessing completed.")


# =========================================================
# 3. EXPLORATORY DATA ANALYSIS
# =========================================================

print("\n========== STATISTICS ==========")
print(df.describe())




# Sales over time
plt.figure(figsize=(12, 5))

plt.plot(
    df["date"],
    df["sales"]
)

plt.xlabel("date")
plt.ylabel("sales")
plt.title("sales Over Time")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# Sales distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["sales"],
    kde=True
)



# =========================================================
# 4. FEATURE ENGINEERING
# =========================================================

df["Year"] = df["date"].dt.year

df["Month"] = df["date"].dt.month

df["Day"] = df["date"].dt.day

df["DayOfWeek"] = df["date"].dt.dayofweek

df["WeekOfYear"] = df["date"].dt.isocalendar().week.astype(int)



# Weekend feature
df["IsWeekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)







print("\n========== FEATURES ==========")

print(df.head())





# =========================================================
# 6. SELECT FEATURES
# =========================================================

features = [
    "store",
    "item",
    "Year",
    "Month",
    "Day",
    "DayOfWeek"
    
]

X = df[features]

y = df["sales"]


print("\nX shape:", X.shape)

print("y shape:", y.shape)


# =========================================================
# 7. TRAIN TEST SPLIT
# =========================================================

# IMPORTANT:
# Since this is forecasting, we don't randomly shuffle
# the data. We train on past data and test on future data.

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print("\n========== TRAIN TEST ==========")

print("X_train:", X_train.shape)

print("X_test :", X_test.shape)

print("y_train:", y_train.shape)

print("y_test :", y_test.shape)


# =========================================================
# 8. LINEAR REGRESSION
# =========================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)


# =========================================================
# 9. RANDOM FOREST
# =========================================================

rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=2
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)


# =========================================================
# 10. XGBOOST
# =========================================================

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    objective="reg:squarederror"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test
)


# =========================================================
# 11. EVALUATION FUNCTION
# =========================================================

def evaluate_model(name, y_test, prediction):

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    mse = mean_squared_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        prediction
    )

    print("\n==============================")

    print(name)

    print("==============================")

    print("MAE :", mae)

    print("RMSE:", rmse)

    print("R2  :", r2)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# =========================================================
# 12. MODEL EVALUATION
# =========================================================

linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    linear_pred
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

xgb_results = evaluate_model(
    "XGBoost",
    y_test,
    xgb_pred
)


# =========================================================
# 13. FIND BEST MODEL
# =========================================================

results = {
    "Linear Regression": linear_results,
    "Random Forest": rf_results,
    "XGBoost": xgb_results
}


best_model_name = max(
    results,
    key=lambda x: results[x]["R2"]
)


print("\n================================")
print("BEST MODEL:", best_model_name)
print("================================")


# Select model
if best_model_name == "Linear Regression":

    best_model = linear_model

elif best_model_name == "Random Forest":

    best_model = rf_model

else:

    best_model = xgb_model


# =========================================================
# 14. SAVE MODEL
# =========================================================

with open("model.pkl", "wb") as file:

    pickle.dump(
        best_model,
        file
    )


# Save product encoder

    


# Save feature names
with open("features.pkl", "wb") as file:

    pickle.dump(
        features,
        file
    )


print("\nModel saved successfully!")

print("model.pkl")
print("product_encoder.pkl")
print("features.pkl")
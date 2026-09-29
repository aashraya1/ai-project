import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend since we only save plots, never display them

# Step: Test that we can load the dataset correctly
import pandas as pd

# Read the CSV file into a pandas DataFrame (like a table)
df = pd.read_csv("dataset/weatherAUS.csv")

# Print basic info to confirm it loaded properly
print("Shape of dataset (rows, columns):", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# ======================================================
# STEP 2: Create the next-day rainfall target + clean data
# ======================================================

# Convert Date column into an actual datetime type (so sorting works correctly)
df['Date'] = pd.to_datetime(df['Date'])

# Sort by Location first, then Date within each location
# This is essential before we shift values — otherwise rows from different
# cities would get mixed together
df = df.sort_values(by=['Location', 'Date'])

# Create the target column: "RainfallTomorrow"
# For each location, shift the Rainfall column up by 1 row.
# This means: today's new column value = tomorrow's actual rainfall amount.
df['RainfallTomorrow'] = df.groupby('Location')['Rainfall'].shift(-1)

# The very last day recorded for each location will now have RainfallTomorrow = NaN
# (there's no "tomorrow" data available for it), so we remove those rows
df = df.dropna(subset=['RainfallTomorrow'])

print("\nShape after creating target and removing last-day-per-location rows:", df.shape)

# ------------------------------------------------------
# Select only the beginner-friendly features we'll use
# ------------------------------------------------------
selected_features = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'WindSpeed9am', 'WindSpeed3pm',
    'Sunshine', 'Cloud9am', 'Cloud3pm', 'RainToday'
]

# Keep only selected features + our new target column
model_df = df[selected_features + ['RainfallTomorrow']].copy()

print("\nShape after selecting relevant columns:", model_df.shape)

# ------------------------------------------------------
# Convert RainToday (Yes/No) into numbers (1/0)
# ------------------------------------------------------
# Machine learning models need numbers, not text.
# Yes -> 1, No -> 0
model_df['RainToday'] = model_df['RainToday'].map({'Yes': 1, 'No': 0})

# ------------------------------------------------------
# Handle missing values
# ------------------------------------------------------
# Check how many missing values each column has before cleaning
print("\nMissing values per column BEFORE filling:")
print(model_df.isnull().sum())

# For the weather measurement columns, fill missing values with the median
# (median is a safe choice — it isn't skewed by extreme outlier values)
for col in selected_features:
    if model_df[col].isnull().sum() > 0:
        median_value = model_df[col].median()
        model_df[col] = model_df[col].fillna(median_value)

# Double check: no missing values should remain
print("\nMissing values per column AFTER filling:")
print(model_df.isnull().sum())

print("\nFinal cleaned dataset shape:", model_df.shape)
print("\nPreview of cleaned data:")
print(model_df.head())


# ======================================================
# STEP 3: Exploratory Data Analysis (EDA) & Visualizations
# ======================================================
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Make sure the graphs folder exists (in case it was deleted)
os.makedirs("graphs", exist_ok=True)

# Set a clean, readable style for all plots
sns.set_style("whitegrid")

# ------------------------------------------------------
# 1. Rainfall distribution
# ------------------------------------------------------
# This shows how rainfall amounts are distributed.
# We expect most values to be near 0 (dry days), with a long tail
# of larger values (heavy rain days) — this is normal for rainfall data.
plt.figure(figsize=(8, 5))
sns.histplot(model_df['RainfallTomorrow'], bins=50, kde=True, color='steelblue')
plt.title("Distribution of Next-Day Rainfall (mm)")
plt.xlabel("Rainfall Tomorrow (mm)")
plt.ylabel("Number of Days")
plt.xlim(0, 100)  # zoom in on the 0-100mm range since extreme outliers skew the view
plt.tight_layout()
plt.savefig("graphs/1_rainfall_distribution.png")
plt.close()
print("Saved: graphs/1_rainfall_distribution.png")

# ------------------------------------------------------
# 2. Monthly rainfall (average per month)
# ------------------------------------------------------
# We use the original 'df' here because it still has the Date column.
df['Month'] = df['Date'].dt.month

monthly_avg = df.groupby('Month')['Rainfall'].mean()

plt.figure(figsize=(8, 5))
monthly_avg.plot(kind='bar', color='cornflowerblue')
plt.title("Average Rainfall by Month (All Locations)")
plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graphs/2_monthly_rainfall.png")
plt.close()
print("Saved: graphs/2_monthly_rainfall.png")

# ------------------------------------------------------
# 3. Humidity vs Rainfall
# ------------------------------------------------------
# We sample 3000 random rows instead of plotting all ~142,000 points
# This keeps the plot fast and readable without losing the overall pattern
sample_df = model_df.sample(3000, random_state=42)

plt.figure(figsize=(8, 5))
sns.scatterplot(data=sample_df, x='Humidity3pm', y='RainfallTomorrow', alpha=0.4, color='teal')
plt.title("Humidity (3pm) vs Next-Day Rainfall")
plt.xlabel("Humidity at 3pm (%)")
plt.ylabel("Rainfall Tomorrow (mm)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("graphs/3_humidity_vs_rainfall.png")
plt.close()
print("Saved: graphs/3_humidity_vs_rainfall.png")

# ------------------------------------------------------
# 4. Temperature vs Rainfall
# ------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(data=sample_df, x='MaxTemp', y='RainfallTomorrow', alpha=0.4, color='darkorange')
plt.title("Max Temperature vs Next-Day Rainfall")
plt.xlabel("Max Temperature (°C)")
plt.ylabel("Rainfall Tomorrow (mm)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("graphs/4_temperature_vs_rainfall.png")
plt.close()
print("Saved: graphs/4_temperature_vs_rainfall.png")

# ------------------------------------------------------
# 5. Correlation heatmap
# ------------------------------------------------------
# This shows how strongly each feature relates to every other feature,
# including our target RainfallTomorrow.
# Values close to +1 or -1 mean a strong relationship; close to 0 means weak/none.
plt.figure(figsize=(10, 8))
correlation_matrix = model_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Weather Features")
plt.tight_layout()
plt.savefig("graphs/5_correlation_heatmap.png")
plt.close()
print("Saved: graphs/5_correlation_heatmap.png")

print("\nAll visualizations saved successfully in the 'graphs' folder!")

# ======================================================
# STEP 4: Split data & train regression models
# ======================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

os.makedirs("models", exist_ok=True)

# ------------------------------------------------------
# Separate features (X) from target (y)
# ------------------------------------------------------
# X = everything the model uses to make a prediction
# y = what the model is trying to predict (tomorrow's rainfall in mm)
X = model_df.drop('RainfallTomorrow', axis=1)
y = model_df['RainfallTomorrow']

# ------------------------------------------------------
# Split into 80% training data, 20% testing data
# ------------------------------------------------------
# random_state=42 makes the split reproducible (same split every time we run this)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# ------------------------------------------------------
# Model 1: Linear Regression
# ------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("\nLinear Regression trained successfully.")

# ------------------------------------------------------
# Model 2: Decision Tree Regression
# ------------------------------------------------------
# max_depth=8 keeps the tree from growing too complex and overfitting
dt_model = DecisionTreeRegressor(max_depth=8, random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree trained successfully.")

# ------------------------------------------------------
# Model 3: Random Forest Regression
# ------------------------------------------------------
# n_estimators=100 means it builds 100 individual decision trees and averages them
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Random Forest trained successfully.")

# ------------------------------------------------------
# Save trained models so we can reuse them later without retraining
# ------------------------------------------------------
joblib.dump(lr_model, "models/linear_regression.pkl")
joblib.dump(dt_model, "models/decision_tree.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")

print("\nAll 3 models trained and saved into the 'models' folder!")



# ======================================================
# STEP 5: Evaluate models & visualize actual vs predicted
# ======================================================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ------------------------------------------------------
# Helper function: calculates and prints all 4 metrics for one model
# ------------------------------------------------------
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Performance:")
    print(f"  MAE  : {mae:.3f} mm")
    print(f"  MSE  : {mse:.3f}")
    print(f"  RMSE : {rmse:.3f} mm")
    print(f"  R2   : {r2:.3f}")

    # Return as a dictionary so we can compare all models later
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# ------------------------------------------------------
# Get predictions from each model on the TEST set
# ------------------------------------------------------
lr_preds = lr_model.predict(X_test)
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# ------------------------------------------------------
# Evaluate all 3 models
# ------------------------------------------------------
results = []
results.append(evaluate_model("Linear Regression", y_test, lr_preds))
results.append(evaluate_model("Decision Tree", y_test, dt_preds))
results.append(evaluate_model("Random Forest", y_test, rf_preds))

# ------------------------------------------------------
# Put results into a comparison table
# ------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n===== Model Comparison Table =====")
print(results_df.to_string(index=False))

# Save comparison table as a CSV too, useful for your report
results_df.to_csv("models/model_comparison.csv", index=False)
print("\nSaved comparison table to models/model_comparison.csv")

# ------------------------------------------------------
# Visualization: Actual vs Predicted (for the best-performing model type: Random Forest)
# ------------------------------------------------------
# We sample 1000 points so the plot stays clean and readable
sample_idx = np.random.RandomState(42).choice(len(y_test), size=1000, replace=False)
y_test_sample = y_test.values[sample_idx]
rf_preds_sample = rf_preds[sample_idx]

plt.figure(figsize=(8, 8))
plt.scatter(y_test_sample, rf_preds_sample, alpha=0.4, color='purple')

# Draw a perfect-prediction diagonal reference line
max_val = max(y_test_sample.max(), rf_preds_sample.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Prediction Line')

plt.title("Actual vs Predicted Rainfall (Random Forest)")
plt.xlabel("Actual Rainfall Tomorrow (mm)")
plt.ylabel("Predicted Rainfall Tomorrow (mm)")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/6_actual_vs_predicted.png")
plt.close()
print("\nSaved: graphs/6_actual_vs_predicted.png")



# ======================================================
# STEP 6: Predict rainfall from user-entered weather values
# ======================================================

def predict_rainfall(min_temp, max_temp, rainfall, humidity_9am, humidity_3pm,
                      pressure_9am, pressure_3pm, wind_speed_9am, wind_speed_3pm,
                      sunshine, cloud_9am, cloud_3pm, rain_today):
    """
    Takes today's weather values and predicts tomorrow's rainfall in mm.
    rain_today should be 1 (Yes) or 0 (No).
    """
    # Build a single-row DataFrame with the SAME column order used in training
    input_data = pd.DataFrame([{
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Sunshine': sunshine,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'RainToday': rain_today
    }])

    # Use the trained Random Forest model (best performer) to predict
    predicted_value = rf_model.predict(input_data)[0]

    # Rainfall can't be negative in real life, so we clip it at 0
    predicted_value = max(0, predicted_value)

    print(f"\nPredicted rainfall: {predicted_value:.2f} mm")
    return predicted_value


# ------------------------------------------------------
# Example usage: a humid, cloudy day (should predict some rain)
# ------------------------------------------------------
print("\n===== Example Prediction 1: Humid, cloudy day =====")
predict_rainfall(
    min_temp=15.0, max_temp=22.0, rainfall=4.0,
    humidity_9am=85, humidity_3pm=78,
    pressure_9am=1012, pressure_3pm=1008,
    wind_speed_9am=15, wind_speed_3pm=20,
    sunshine=2.0, cloud_9am=7, cloud_3pm=8,
    rain_today=1
)

# ------------------------------------------------------
# Example usage: a dry, sunny day (should predict little/no rain)
# ------------------------------------------------------
print("\n===== Example Prediction 2: Dry, sunny day =====")
predict_rainfall(
    min_temp=18.0, max_temp=30.0, rainfall=0.0,
    humidity_9am=40, humidity_3pm=30,
    pressure_9am=1020, pressure_3pm=1018,
    wind_speed_9am=10, wind_speed_3pm=12,
    sunshine=10.5, cloud_9am=1, cloud_3pm=1,
    rain_today=0
)



# ======================================================
# STEP 7: Interactive prediction (user enters their own values)
# ======================================================

def get_float_input(prompt):
    """Helper function: keeps asking until the user enters a valid number."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def interactive_prediction():
    print("\n===== Enter Today's Weather to Predict Tomorrow's Rainfall =====")

    min_temp = get_float_input("Min Temperature (°C): ")
    max_temp = get_float_input("Max Temperature (°C): ")
    rainfall = get_float_input("Today's Rainfall (mm): ")
    humidity_9am = get_float_input("Humidity at 9am (%): ")
    humidity_3pm = get_float_input("Humidity at 3pm (%): ")
    pressure_9am = get_float_input("Pressure at 9am (hPa): ")
    pressure_3pm = get_float_input("Pressure at 3pm (hPa): ")
    wind_speed_9am = get_float_input("Wind Speed at 9am (km/h): ")
    wind_speed_3pm = get_float_input("Wind Speed at 3pm (km/h): ")
    sunshine = get_float_input("Sunshine (hours): ")
    cloud_9am = get_float_input("Cloud cover at 9am (0-9 oktas): ")
    cloud_3pm = get_float_input("Cloud cover at 3pm (0-9 oktas): ")

    rain_today_input = input("Did it rain today? (yes/no): ").strip().lower()
    rain_today = 1 if rain_today_input == "yes" else 0

    predict_rainfall(
        min_temp, max_temp, rainfall, humidity_9am, humidity_3pm,
        pressure_9am, pressure_3pm, wind_speed_9am, wind_speed_3pm,
        sunshine, cloud_9am, cloud_3pm, rain_today
    )


# ------------------------------------------------------
# Ask if the user wants to try their own prediction
# ------------------------------------------------------
try_it = input("\nWould you like to enter your own weather values for a prediction? (yes/no): ").strip().lower()
if try_it == "yes":
    interactive_prediction()
else:
    print("Skipping interactive prediction. Project run complete.")
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("usage_sample.csv", parse_dates=["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

# Create lag features
df["lag1"] = df["cpu"].shift(1)
df["lag24"] = df["cpu"].shift(24)
df = df.dropna()

# Define features and target
X = df[["lag1", "lag24"]]
y = df["cpu"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.joblib")
print("Model saved!")

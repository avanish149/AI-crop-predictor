import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

# Validate Python runtime version
if sys.version_info < (3, 8):
    print("Warning: Python 3.8+ is recommended. Current:", sys.version)
    print("Running with Python:", sys.version.split()[0])

# Dataset configuration
DATAFILE = "crop_recommendation.csv"
print(f"\nSearching for dataset file: {DATAFILE}")
if not os.path.isfile(DATAFILE):
    print(f"ERROR: Dataset file '{DATAFILE}' not found in working directory:\n {os.getcwd()}")
    sys.exit(1)

# Load dataset
try:
    data = pd.read_csv(DATAFILE)
    print("Dataset loaded successfully. Shape:", data.shape)
except Exception:
    print("ERROR: Failed to read CSV file.")
    raise

# Standardize column names (supports alternate headers)
column_map = {
    "Nitrogen": "N",
    "Phosphorus": "P",
    "Potassium": "K",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "pH_Value": "ph",
    "Rainfall": "rainfall",
    "Crop": "label"
}
data = data.rename(columns=column_map)

# Map crop-specific yield (kg/ha) and price (INR/kg)
crop_data = {
    'rice':        (3850, 25.5),
    'maize':       (4200, 18.2),
    'chickpea':    (850,  65.0),
    'kidneybeans': (2800, 45.0),
    'pigeonpeas':  (720,  70.0),
    'mothbeans':   (450,  55.0),
    'mungbean':    (500,  60.0),
    'blackgram':   (480,  58.0),
    'lentil':      (950,  62.0),
    'pomegranate': (22000, 80.0),
    'banana':      (35000, 35.0),
    'mango':       (8500, 45.0),
    'grapes':      (22000, 120.0),
    'watermelon':  (25000, 12.0),
    'muskmelon':   (28000, 15.0),
    'apple':       (20000, 150.0),
    'orange':      (15000, 40.0),
    'papaya':      (35000, 25.0),
    'coconut':     (14000, 30.0),
    'cotton':      (800,  120.0),
    'jute':        (2500, 35.0),
    'coffee':      (1200, 200.0)
}

# Ensure label is lowercase to match dictionary keys
data["label"] = data["label"].str.strip().str.lower()

# Add crop-specific yield and rate columns
yield_rate_df = data["label"].map(crop_data).apply(pd.Series)
yield_rate_df.columns = ["yield", "rates"]
data = pd.concat([data, yield_rate_df], axis=1)

# Validate required columns
required_columns = {
    "N", "P", "K", "temperature", "humidity",
    "ph", "rainfall", "label", "rates", "yield"
}
missing = required_columns - set(data.columns)
if missing:
    print(f"ERROR: Dataset is missing required columns: {missing}")
    sys.exit(1)

print("Column validation successful. Using the following columns:")
print(list(data.columns))

# Scale monetary and yield features to a common range
scaler = MinMaxScaler()
data[["rates", "yield"]] = scaler.fit_transform(data[["rates", "yield"]])

# Split into features and target
X = data.drop("label", axis=1)
y = data["label"]

print("\nFeature data types:")
print(X.dtypes)

# Train/validation split and model training
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nTraining RandomForestClassifier (this may take a few seconds)...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model training complete. Test accuracy: {acc*100:.2f}%")
except Exception:
    print("ERROR: Exception during training or evaluation.")
    raise

# Persist trained model to disk
outname = "crop_model.pkl"
try:
    with open(outname, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to: {outname}")
except Exception:
    print("ERROR: Failed to serialize model.")
    raise

# Convenience wrapper for single-sample prediction
def predict_crop(N, P, K, temperature, humidity, ph, rainfall,
                 rates=25.0, crop_yield=3000.0):
    """Return the predicted crop for a single set of agronomic conditions."""
    df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall, rates, crop_yield]],
        columns=["N", "P", "K", "temperature", "humidity",
                 "ph", "rainfall", "rates", "yield"]
    )
    return model.predict(df)[0]

print("\nRunning demo prediction with example inputs:")
try:
    demo = predict_crop(42, 0, 0, 25, 80, 6.5, 200)
    print("Predicted crop:", demo)
except Exception:
    print("Demo prediction failed.")
    raise

print("\nPipeline completed successfully.")




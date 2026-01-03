# DNN Classification on ThingSpeak Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_NAME = "feed.csv.csv" # Ensure this file is in the same directory as the script!
ANOMALY_THRESHOLD = 150  # Sensor reading in 'field1' above this value will be labeled '1' (Anomaly)
EPOCHS = 30
BATCH_SIZE = 8
# ---------------------

# STEP 1: Load the CSV file
try:
    data = pd.read_csv(FILE_NAME)
    print("Data Loaded Successfully!") 
    print(data.head())
except FileNotFoundError:
    print(f"ERROR: File '{FILE_NAME}' not found. Please check the file name and location.")
    exit()

# STEP 2: Smart Cleaning and Preprocessing
initial_rows = len(data)

# Drop columns that are metadata or typically empty (e.g., location data, and the empty 'status' column)
# We drop 'status' here because it was likely empty and caused the previous error.
columns_to_drop = ['created_at', 'entry_id', 'latitude', 'longitude', 'elevation', 'status']
data = data.drop(columns_to_drop, axis=1, errors='ignore')

# Remove any rows where sensor data (fieldN) are missing
data = data.dropna()  

rows_after_cleaning = len(data)
if data.empty:
    print("\n--- Data Error ---")
    print(f"WARNING: All {initial_rows} rows were removed after cleaning.")
    print("This means your sensor fields (e.g., 'field1') are entirely empty (NaN).")
    print("Please check your ThingSpeak channel data for sensor readings.")
    exit()
else:
    print(f"\nData Cleaning Complete: Retained {rows_after_cleaning} rows (dropped {initial_rows - rows_after_cleaning} rows).")


# STEP 3: Define Features (X) and TEMPORARY Target (Y)
# We must create a target 'y' since your CSV file did not contain labels.
# This creates a binary classification problem: Anomaly (1) vs. Normal (0).

# X (Features) are all remaining columns (e.g., field1, field2, etc.)
X = data.values 

# Y (Target) is created by classifying data based on a threshold on 'field1'
# This assumes 'field1' is the first column remaining in the DataFrame after cleaning.
if 'field1' in data.columns:
    # Create the binary label: 1 if field1 value >= threshold, 0 otherwise.
    y = (data['field1'] >= ANOMALY_THRESHOLD).astype(int).values
    print(f"\nTarget (Y) created based on 'field1' where >= {ANOMALY_THRESHOLD} is classified as ANOMALY (1).")
else:
    print("\nERROR: Could not find 'field1' column to create the target variable. Please adjust the code.")
    exit()
    
# STEP 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Normalize Data (Crucial for DNN performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 6: Build the DNN Model
# Model is configured for BINARY Classification (Output layer uses 1 unit and 'sigmoid' activation)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# STEP 7: Train the Model
print("\n--- Training Model ---")
# Suppress a TensorFlow warning about casting to float32
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

history = model.fit(
    X_train, 
    y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_split=0.2, 
    verbose=1
)

# STEP 8: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# FIX: Removed the emoji from the print statement to avoid UnicodeEncodeError
print(f"\n--- DNN Model Accuracy on Test Set: {accuracy*100:.2f}%")

# STEP 9: Plot Accuracy & Loss Graphs
plt.figure(figsize=(10,4))

# Plot Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

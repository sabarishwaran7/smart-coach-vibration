# smart-coach-vibration

A simple Deep Neural Network (DNN) for detecting anomalies in ThingSpeak IoT sensor data.  
This project loads a ThingSpeak CSV export, cleans and normalizes sensor fields, creates binary labels (anomaly vs normal) from `field1`, trains a DNN, and evaluates accuracy.

## Files
- dnn_classification.py — main training/evaluation script
- tempCodeRunnerFile.py — helper / backup script
- feed.csv.csv — ThingSpeak CSV export (input data)

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- tensorflow (or tensorflow-cpu)
- matplotlib

Install:

## Configuration
Edit the top of `dnn_classification.py`:
- FILE_NAME — CSV filename (default: feed.csv.csv)
- ANOMALY_THRESHOLD — numeric threshold for labeling anomalies
- EPOCHS — training epochs
- BATCH_SIZE — training batch size

## Usage
Run the script from the project directory:

## Workflow
1. Load CSV and drop metadata columns (created_at, entry_id, latitude, longitude, elevation, status).
2. Drop rows with missing sensor values.
3. Use remaining fields as features; generate binary target from `field1` using ANOMALY_THRESHOLD.
4. Split 80/20 train/test, scale features with StandardScaler.
5. Train a 3-layer DNN (Dense 64 → Dense 32 → Dense 1 with sigmoid).
6. Evaluate on test set and print loss/accuracy.

## Notes & Troubleshooting
- Ensure `feed.csv.csv` contains sensor columns named like `field1`, `field2`, etc.
- If data is removed entirely during cleaning, check ThingSpeak channel for missing values.
- For authentication or git issues, set up a GitHub PAT or SSH key when pushing changes.

## License
Add your preferred license text or leave unspecified.

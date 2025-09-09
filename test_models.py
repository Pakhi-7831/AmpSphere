import pandas as pd
import numpy as np
import joblib
from datetime import datetime

print("Testing Trained Models")
print("=" * 30)
try:
    soc_model = joblib.load('models/soc_model.pkl')
    soh_model = joblib.load('models/soh_model.pkl')
    risk_model = joblib.load('models/risk_model.pkl')
    alert_model = joblib.load('models/alert_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    print("All models loaded successfully!")
    
except FileNotFoundError as e:
    print(f" Model file not found: {e}")
    print(" Make sure you ran train_battery_models.py first")
    exit()

try:
    df = pd.read_csv('data/processed_dataset_with_targets.csv')
    print(f" Loaded test dataset: {len(df)} rows")
except FileNotFoundError:
    print(" Processed dataset not found")
    exit()

feature_cols = [
    'Battery_Age_Months', 'Avg_Cell_Voltage', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
    'Pack_Current', 'Avg_Temperature', 'Max_Temperature', 'Min_Temperature',
    'Internal_Resistance', 'Pressure_Level', 'Coolant_Flow_Rate', 'Cycle_Count',
    'Voltage_Imbalance', 'Temperature_Gradient', 'Power_kW',
    'Humidity_Percent', 'Air_Pressure_kPa', 'Hour', 'Day_of_Week', 'Month',
    'Vehicle_Model_Encoded', 'Charging_State_Encoded', 'Location_City_Encoded'
]

# Get test sample
test_sample = df[feature_cols].iloc[0:5]  # First 5 rows
test_sample_scaled = scaler.transform(test_sample)

print(f" Testing with {len(test_sample)} sample predictions...")

soc_predictions = soc_model.predict(test_sample_scaled)
print(f"\n SOC Predictions:")
for i, pred in enumerate(soc_predictions):
    actual = df['SOC'].iloc[i]
    print(f"  Sample {i+1}: Predicted={pred:.1f}%, Actual={actual:.1f}%, Error={abs(pred-actual):.1f}%")

# Predict SOH
soh_predictions = soh_model.predict(test_sample_scaled)
print(f"\n SOH Predictions:")
for i, pred in enumerate(soh_predictions):
    actual = df['SOH'].iloc[i]
    print(f"  Sample {i+1}: Predicted={pred:.1f}%, Actual={actual:.1f}%, Error={abs(pred-actual):.1f}%")

# Predict Risk
risk_predictions = risk_model.predict(test_sample_scaled)
print(f"\n Risk Predictions:")
for i, pred in enumerate(risk_predictions):
    risk_score = df['Overall_Risk_Score'].iloc[i]
    print(f"  Sample {i+1}: Predicted={pred}, Risk Score={risk_score:.1f}")

# Predict Alerts
alert_predictions = alert_model.predict(test_sample_scaled)
print(f"\n Alert Predictions:")
for i, pred in enumerate(alert_predictions):
    actual_alert = df['Alert_Status'].iloc[i]
    pred_text = "CRITICAL" if pred == 1 else "NORMAL"
    print(f"  Sample {i+1}: Predicted={pred_text}, Actual={actual_alert}")

def predict_single_battery(battery_data):
    """
    Make prediction for a single battery reading
    
    Args:
        battery_data (dict): Dictionary with sensor readings
        
    Returns:
        dict: Predictions for SOC, SOH, risk, alerts
    """
    
    required_fields = ['Avg_Cell_Voltage', 'Pack_Current', 'Avg_Temperature', 'Battery_Age_Months']
    for field in required_fields:
        if field not in battery_data:
            raise ValueError(f"Missing required field: {field}")
    
    sample_features = test_sample.iloc[0].copy()
    
    for key, value in battery_data.items():
        if key in sample_features.index:
            sample_features[key] = value
    
    features_scaled = scaler.transform([sample_features.values])
    
    soc_pred = soc_model.predict(features_scaled)[0]
    soh_pred = soh_model.predict(features_scaled)[0]
    risk_pred = risk_model.predict(features_scaled)[0]
    alert_pred = alert_model.predict(features_scaled)[0]
    
    return {
        'timestamp': datetime.now().isoformat(),
        'predictions': {
            'SOC': round(soc_pred, 1),
            'SOH': round(soh_pred, 1),
            'risk_level': risk_pred,
            'critical_alert': bool(alert_pred),
            'alert_status': 'CRITICAL' if alert_pred else 'NORMAL'
        },
        'input_data': battery_data
    }

print(f"\n Demo Prediction:")
print("-" * 20)

demo_data = {
    'Avg_Cell_Voltage': 3.65,
    'Pack_Current': 45.2,
    'Avg_Temperature': 32.5,
    'Battery_Age_Months': 24,
    'Internal_Resistance': 3.1,
    'Cycle_Count': 450
}

try:
    demo_result = predict_single_battery(demo_data)
    
    print(f" Input Data:")
    for key, value in demo_data.items():
        print(f"  {key}: {value}")
    
    print(f"\n Predictions:")
    preds = demo_result['predictions']
    print(f"  SOC: {preds['SOC']}%")
    print(f"  SOH: {preds['SOH']}%")
    print(f"  Risk Level: {preds['risk_level']}")
    print(f"  Alert Status: {preds['alert_status']}")
    
except Exception as e:
    print(f" Prediction error: {e}")

print(f"\n Model Statistics:")
print("-" * 20)

soc_errors = abs(soc_predictions - df['SOC'].iloc[0:5])
soh_errors = abs(soh_predictions - df['SOH'].iloc[0:5])

print(f"SOC Model:")
print(f"  Average Error: ±{soc_errors.mean():.2f}%")
print(f"  Max Error: ±{soc_errors.max():.2f}%")

print(f"SOH Model:")
print(f"  Average Error: ±{soh_errors.mean():.2f}%")
print(f"  Max Error: ±{soh_errors.max():.2f}%")

print(f"\n Model testing completed!")
print(f" Models are ready for real-time predictions!")

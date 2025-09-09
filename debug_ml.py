import os
import sys
from pathlib import Path

print(" DEBUGGING ML SETUP")
print("=" * 30)

# Check current directory
print(f" Current directory: {os.getcwd()}")
print(f" Files in current directory:")
for item in os.listdir('.'):
    print(f"   {item}")

print("\n" + "="*30)

directories_to_check = ['data', 'models']
for dir_name in directories_to_check:
    if os.path.exists(dir_name):
        print(f" {dir_name}/ directory exists")
        files = os.listdir(dir_name)
        if files:
            print(f"   Files: {files}")
        else:
            print(f"     Directory is empty")
    else:
        print(f" {dir_name}/ directory missing")

print("\n" + "="*30)

dataset_files = [
    'data/raw_battery_dataset_1000.csv',
    'data/processed_battery_data.csv',
    'data/processed_dataset_with_targets.csv'
]

for file_path in dataset_files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f" {file_path} exists ({size:,} bytes)")
    else:
        print(f" {file_path} not found")

print("\n" + "="*30)

model_files = [
    'models/soc_model.pkl',
    'models/soh_model.pkl', 
    'models/risk_model.pkl',
    'models/alert_model.pkl',
    'models/scaler.pkl',
    'models/simple_battery_models.pkl'
]

models_found = 0
for file_path in model_files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f" {file_path} exists ({size:,} bytes)")
        models_found += 1
    else:
        print(f" {file_path} not found")

print(f"\n Found {models_found} model files")

print("\n" + "="*30)

packages_to_check = ['pandas', 'numpy', 'sklearn', 'joblib', 'pickle']
for package in packages_to_check:
    try:
        if package == 'sklearn':
            import sklearn
            print(f" {package} imported successfully")
        elif package == 'pickle':
            import pickle
            print(f" {package} imported successfully")
        else:
            exec(f"import {package}")
            print(f" {package} imported successfully")
    except ImportError:
        print(f" {package} not installed")

print("\n" + "="*30)

print(" NEXT STEPS:")

if models_found == 0:
    print(" No models found!")
    print(" You need to run training first:")
    print("   1. Make sure your dataset is in data/")
    print("   2. Run: python train_battery_models.py")
    print("   3. Then run: python test_models.py")
elif models_found < 4:
    print("  Some models missing!")
    print(" Try re-running training:")
    print("   python train_battery_models.py")
else:
    print(" Models found! Test script should work.")
    print(" Try running:")
    print("   python test_models.py")

print("\n If Thonny opens instead of command line:")
print("   1. Open Command Prompt (cmd)")
print("   2. Navigate to your project folder:")
print("   3. Run: python debug_ml.py")
print("   4. Then: python test_models.py")

# Test basic functionality
print("\n BASIC FUNCTIONALITY TEST:")
try:
    import pandas as pd
    print(" pandas works")
    
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        test_file = csv_files[0]
        df = pd.read_csv(test_file)
        print(f" Can read CSV files: {test_file} ({len(df)} rows)")
    else:
        print("  No CSV files found to test")
        
except Exception as e:
    print(f" Basic functionality failed: {e}")

print("\n" + "="*30)
print(" Status check complete!")

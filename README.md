# 🔋 AmpSphere: Battery Health Prediction Dashboard

Welcome to the AmpSphere project! ⚡️

This application uses machine learning to analyze battery operational data and predict its State of Health (SoH). It provides an interactive dashboard built with Streamlit to help you assess battery degradation and identify potential issues before they become critical failures.



---

## ✨ Features

* **Health Prediction**: Upload your battery data (CSV) and get an instant prediction of its health status.
* **Interactive Dashboard**: A simple, user-friendly interface for visualizing results.
* **Critical Level Assessment**: Helps flag batteries that may require maintenance.

---

## 🛠️ Tech Stack

* **Web Framework**: Streamlit
* **Machine Learning**: Scikit-learn, Joblib
* **Data Handling**: Pandas, NumPy
* **Development**: Jupyter Notebook

---

## Getting Started

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  Clone the repository:

        git clone https://github.com/your-username/jlr-battery-health-platform.git
    cd jlr-battery-health-platform
    

2.  Create and activate a virtual environment:

        python -m venv .venv
    source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
    

3.  Install the required packages:

        pip install -r jlr-battery-health-platform/requirements.txt
    

### Running the Application

1.  Run the Streamlit dashboard:

        streamlit run jlr-battery-health-platform/dashboard/app.py
    

2.  Open your browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

## Project Structure

jlr-battery-health-platform/
├── dashboard/
│   └── app.py              # Streamlit dashboard application
├── data/
│   ├── processed_dataset_with_targets.csv
│   └── raw_battery_dataset_1000.csv
├── models/
│   ├── alert_model.pkl
│   ├── feature_importance.csv
│   ├── risk_model.pkl
│   ├── scaler.pkl
│   ├── soc_model.pkl
│   ├── soh_model.pkl
│   ├── training_battery_models.py
│   └── encoders/
│       ├── charging_state_encoder.pkl
│       ├── location_encoder.pkl
│       └── vehicle_model_encoder.pkl
├── debug_ml.py
├── generator.py
├── installed_packages.txt
├── requirements.txt
└── test_models.py

## Dependencies

The main dependencies are listed in jlr-battery-health-platform/requirements.txt and include:

*   streamlit
*   pandas
*   numpy
*   plotly
*   scikit-learn
*   joblib

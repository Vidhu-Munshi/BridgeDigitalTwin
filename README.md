# Bridge Digital Twin

### Real-Time Structural Health Monitoring System

Bridge Digital Twin is a real-time smart infrastructure monitoring system designed to assess bridge safety using multi-sensor data and machine learning.

The system integrates Arduino-based data acquisition with a Python backend and a Random Forest model to predict the **Structural Health Index (SHI)**. It combines data-driven intelligence with rule-based logic to detect multiple risk conditions in real time.

---

## Features
- Real-time sensor data acquisition (Arduino)
- SHI prediction using Random Forest (ML)
- Multi-risk detection:
  - SAFE
  - STRUCTURAL DISTURBANCE
  - FIRE INCIDENT
  - CRITICAL CONDITION
- Live dashboard with graphical visualization
- Lightweight and real-time processing

---

## Dataset
Kaggle Dataset Link:  
https://www.kaggle.com/datasets/mithil27360/digital-twin-bridge-structural-health-monitoring  

Download the dataset and place it in the project directory before running the model.

---

## Sensors Used
- **Piezoelectric Sensor (Analog)** → Measures vibration intensity  
- **Digital Vibration Sensor (0/1)** → Detects sudden disturbances  
- **Fire Sensor (Analog)** → Detects heat/flame conditions  

---

## System Architecture
- **Sensors**  
  Piezoelectric, Vibration, and Fire sensors collect real-time physical data from the bridge.

- **Arduino**  
  Reads sensor values and transmits data via serial communication.

- **Serial Communication**  
  Connects Arduino with the Python backend.

- **Python Backend (Flask)**  
  Processes incoming data and prepares it for prediction.

- **ML Model (Random Forest)**  
  Predicts Structural Health Index (SHI).

- **Dashboard (Frontend)**  
  Displays SHI values, alerts, and system status in real time.

---

## Core Concept
- Machine Learning → SHI prediction  
- Rule-based logic → Real-time hazard detection  
- Hybrid system → Accuracy + reliability  

---

## Tech Stack
- Arduino (C/C++)
- Python (Flask)
- Scikit-learn (Random Forest)
- HTML / CSS / JavaScript

---

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Vidhu-Munshi/BridgeDigitalTwin.git
cd BridgeDigitalTwin
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Connect Arduino and configure the correct COM port  

4. Run the backend:
```bash
python app.py
```

---

## Frontend Note
- `gov.html` and `public.html` are hardcoded prototype files  
- These are created only for UI demonstration  
- Not fully integrated with backend logic  

---

## Disclaimer
This is a prototype system and is not intended for real-world deployment without further testing and validation.

---

## Tagline
> Turning raw sensor signals into intelligent infrastructure insights.

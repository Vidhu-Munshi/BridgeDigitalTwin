# Bridge Digital Twin

### Real-Time Structural Health Monitoring System

Bridge Digital Twin is a real-time smart infrastructure monitoring system designed to assess bridge safety using multi-sensor data and machine learning.

The system integrates Arduino-based data acquisition with a Python backend and a Random Forest model to predict the **Structural Health Index (SHI)**. It combines data-driven intelligence with rule-based logic to detect multiple risk conditions in real time.
## Features
-  Real-time sensor data acquisition (Arduino)
-  SHI prediction using Random Forest (ML)
-  Multi-risk detection:
  - SAFE
  - STRUCTURAL DISTURBANCE
  - FIRE INCIDENT
  - CRITICAL CONDITION
-  Live dashboard with graphical visualization
-  Lightweight and real-time processing
## Sensors Used
- **Piezoelectric Sensor (Analog)** → Measures vibration intensity  
- **Digital Vibration Sensor (0/1)** → Detects sudden disturbances  
- **Fire Sensor (Analog)** → Detects heat/flame conditions  
## System Architectur
- **Sensors**  
  Piezoelectric, Vibration, and Fire sensors collect real-time physical data from the bridge.
- **Arduino**  
  Reads sensor values and transmits structured data via serial communication.
- **Serial Communication**  
  Acts as a bridge between hardware (Arduino) and software (Python).
- **Python Backend**  
  Processes incoming sensor data, applies logic, and prepares it for analysis.
- **ML Model (Random Forest)**  
  Predicts the Structural Health Index (SHI) based on sensor input.
- **Dashboard**  
  Displays real-time system status, SHI values, and alerts with graphical visualization.
## Core Concept
- Machine Learning is used for **Structural Health Index (SHI) prediction**
- Rule-based logic is used for **real-time hazard detection**
- Hybrid approach ensures **accuracy + reliability**
## Tech Stack
- Arduino (C/C++)
- Python (Flask)
- Scikit-learn (Random Forest)
- HTML/CSS/JS (Frontend)
## Note
This project demonstrates a practical implementation of a **Digital Twin for infrastructure monitoring**, combining real-time data processing with intelligent decision-making.
## Tagline
> Turning raw sensor signals into intelligent infrastructure insights.

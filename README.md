# S.A.H.A.R.A - Smart Surveillance and Public Safety Platform

S.A.H.A.R.A is an AI-powered surveillance system designed to detect and alert authorities about public safety threats in real-time. Using advanced machine learning models (YOLOv8, VideoMAE, and Roboflow), it monitors live video feeds to identify instances of vandalism, violence, and spitting.

## 🚀 Features

- **Real-Time Video Analytics:** Processes live webcam/CCTV feeds instantly.
- **Vandalism Detection:** Uses YOLOv8 to detect objects associated with vandalism (e.g., spray cans, baseball bats, knives).
- **Violence Detection:** Uses VideoMAE (Video Masked Autoencoders) for action recognition to detect fighting, punching, and kicking.
- **Spitting Detection:** Integrates with Roboflow's cloud API for highly accurate spitting detection.
- **Live Dashboard:** A modern, premium dark-mode web interface built with Vite and React/Vanilla JS to monitor the live feed and view alerts.
- **Instant Alerts:** Real-time WebSocket notifications instantly log incidents to the web dashboard.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, OpenCV, PyTorch, Ultralytics (YOLO), Transformers (HuggingFace)
- **Frontend:** HTML, CSS (Glassmorphism UI), Vanilla JS, WebSockets
- **Models:**
  - YOLOv8 (Local Object Detection)
  - VideoMAE-base-finetuned-kinetics (Local Action Recognition)
  - Roboflow Custom Model (Cloud Object Detection)

## 📦 Installation & Setup

### 1. Backend Setup

Ensure you have Python 3.8+ installed.

```bash
# Navigate to the backend directory
cd backend

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

You must place the `yolov8l.pt` weights file in the root directory (outside the backend folder) for the vandalism detection to work.

### 2. Frontend Setup

Ensure you have Node.js and NPM installed.

```bash
# Navigate to the frontend directory
cd frontend

# Install Node modules
npm install
```

## 🚀 Running the System

You will need two separate terminal windows to run both the API and the Dashboard.

**Terminal 1: Start the Backend**
```bash
cd backend
python app.py
```
*The server will start on `http://127.0.0.1:8000` and connect to your webcam.*

**Terminal 2: Start the Dashboard**
```bash
cd frontend
npm run dev
```
*Open the provided localhost link (usually `http://localhost:5173`) in your browser to view the S.A.H.A.R.A dashboard.*

## ⚠️ Hardware Integration (Upcoming)

S.A.H.A.R.A is designed with hardware in mind. The `backend/app.py` contains placeholders to easily trigger physical alarms, sirens, or LEDs via Raspberry Pi GPIO pins whenever a threat is detected.

## 📄 License

This project is open-source and available under the MIT License.

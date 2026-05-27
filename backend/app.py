import cv2
import torch
import time
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from collections import deque
from roboflow import Roboflow

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
ALARM_COOLDOWN = 10.0 
last_triggered_time = 0
active_websockets = set()
frames_queue = deque(maxlen=16)

# --- LOAD MODELS ---
print("Loading all S.A.H.A.R.A. models...")
vandalism_model = YOLO('../yolov8l.pt')
VANDALISM_OBJECTS = ['spray can', 'knife', 'baseball bat']

violence_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
violence_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
VIOLENCE_KEYWORDS = ["fighting", "punching", "kicking", "wrestling", "shooting"]

ROBOFLOW_API_KEY = "3EnjAELcduturMzqpye3"
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    workspace = rf.workspace("ecs-project-3noyw")
    project = workspace.project("spit-detection-nlkqi-egijm")
    spitting_model = project.version(2).model
    print("Spitting model connected.")
except Exception as e:
    print(f"Could not connect to Spitting model. {e}")
    spitting_model = None

# --- NOTIFICATION SYSTEM ---
async def broadcast_alert(threat_type):
    global last_triggered_time
    current_time = time.time()
    
    if (current_time - last_triggered_time) > ALARM_COOLDOWN:
        print(f"ALARM TRIGGERED: {threat_type}")
        
        # >>> TODO: Email/SMS or Hardware GPIO Trigger here <<<
        
        message = json.dumps({
            "type": "alert",
            "threat": threat_type,
            "timestamp": current_time
        })
        
        # Broadcast to all connected frontend clients
        for ws in list(active_websockets):
            try:
                await ws.send_text(message)
            except Exception:
                active_websockets.remove(ws)
                
        last_triggered_time = current_time

# --- VIDEO GENERATOR ---
def generate_frames():
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Task 1: Vandalism (Local)
        vandalism_results = vandalism_model(frame, verbose=False)
        for box in vandalism_results[0].boxes:
            if box.conf[0] > 0.6:
                class_name = vandalism_model.names[int(box.cls[0])]
                if class_name in VANDALISM_OBJECTS:
                    # Run alert async without blocking video
                    asyncio.run(broadcast_alert(f"Vandalism Risk ({class_name})"))
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, f"VANDALISM: {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Task 2: Spitting (Cloud)
        if spitting_model:
            try:
                spitting_results = spitting_model.predict(frame, confidence=50, overlap=50).json()
                for prediction in spitting_results['predictions']:
                    asyncio.run(broadcast_alert("Spitting"))
                    x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "SPITTING", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            except Exception:
                pass

        # Task 3: Violence (Local)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_queue.append(rgb_frame)
        if len(frames_queue) == 16:
            inputs = violence_processor(list(frames_queue), return_tensors="pt")
            with torch.no_grad():
                outputs = violence_model(**inputs)
                predicted_label = violence_model.config.id2label[outputs.logits.argmax(-1).item()]
            if any(keyword in predicted_label for keyword in VIOLENCE_KEYWORDS):
                asyncio.run(broadcast_alert(f"Violence ({predicted_label})"))
                cv2.putText(frame, f"VIOLENCE: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    video_capture.release()

@app.get("/video_feed")
def video_feed():
    """Stream MJPEG video to the frontend."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/alerts")
async def alerts_websocket(websocket: WebSocket):
    """WebSocket for real-time threat alerts."""
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception:
        active_websockets.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

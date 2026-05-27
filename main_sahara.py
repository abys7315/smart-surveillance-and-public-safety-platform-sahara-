import cv2
import torch
from ultralytics import YOLO
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from collections import deque
import time
from roboflow import Roboflow

# --- 1. CONFIGURATION ---
ALARM_COOLDOWN = 10.0 
last_triggered_time = 0

# --- 2. HARDWARE CONTROL FUNCTION (PLACEHOLDER) ---
def trigger_alarm(threat_type):
    global last_triggered_time
    current_time = time.time()
    
    if (current_time - last_triggered_time) > ALARM_COOLDOWN:
        print(f"========================================")
        print(f"ALARM TRIGGERED!")
        print(f"Threat Detected: {threat_type}")
        print(f"========================================")
        
        # >>> THIS IS WHERE YOUR RASPBERRY PI GPIO CODE GOES <<<
        
        last_triggered_time = current_time

# --- 3. LOAD ALL ML MODELS ---
print("Loading all S.A.H.A.R.A. models...")

# --- A. Load LOCAL Models ---
print("-> Loading Vandalism Detection Model (Local)...")
vandalism_model = YOLO('yolov8l.pt')
VANDALISM_OBJECTS = ['spray can', 'knife', 'baseball bat']
print("   Vandalism model loaded.")

print("-> Loading Violence Detection Model (Local)...")
violence_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
violence_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
frames_queue = deque(maxlen=16)
VIOLENCE_KEYWORDS = ["fighting", "punching", "kicking", "wrestling", "shooting"]
print(" Violence model loaded.")

# --- B. Connect to CLOUD Model (Roboflow API) ---
print("-> Connecting to Spitting Detection Model (Cloud)...")
# 🛑 PASTE YOUR NEWEST, SECRET API KEY HERE
ROBOFLOW_API_KEY = "3EnjAELcduturMzqpye3"
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    workspace = rf.workspace("ecs-project-3noyw")
    project = workspace.project("spit-detection-nlkqi-egijm")
    spitting_model = project.version(2).model
    print("   Spitting model connected.")
except Exception as e:
    print(f"  Could not connect to Spitting model. Please check API Key. Exiting.")
    exit()

print("\nSystem armed. Starting real-time monitoring...")

# --- 4. INITIALIZE VIDEO CAPTURE ---
video_capture = cv2.VideoCapture(0)

# --- 5. MAIN DETECTION LOOP ---
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # --- Task 1: Vandalism Detection (Local) ---
    vandalism_results = vandalism_model(frame, verbose=False)
    for box in vandalism_results[0].boxes:
        if box.conf[0] > 0.6:
            class_name = vandalism_model.names[int(box.cls[0])]
            if class_name in VANDALISM_OBJECTS:
                trigger_alarm(f"Vandalism Risk ({class_name})")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, f"VANDALISM: {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # --- Task 2: Spitting Detection (Cloud API) ---
    spitting_results = spitting_model.predict(frame, confidence=50, overlap=50).json()
    for prediction in spitting_results['predictions']:
        trigger_alarm("Spitting")
        x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue box
        cv2.putText(frame, "SPITTING", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    # --- Task 3: Violence Detection (Local) ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames_queue.append(rgb_frame)
    if len(frames_queue) == 16:
        inputs = violence_processor(list(frames_queue), return_tensors="pt")
        with torch.no_grad():
            outputs = violence_model(**inputs)
            predicted_label = violence_model.config.id2label[outputs.logits.argmax(-1).item()]
        if any(keyword in predicted_label for keyword in VIOLENCE_KEYWORDS):
            trigger_alarm(f"Violence ({predicted_label})")
            cv2.putText(frame, f"VIOLENCE: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Status: Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- 6. DISPLAY THE COMBINED FRAME ---
    cv2.imshow('S.A.H.A.R.A. Main System (Press "q" to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. CLEANUP ---
video_capture.release()
cv2.destroyAllWindows()
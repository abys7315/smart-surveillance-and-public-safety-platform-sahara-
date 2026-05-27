import cv2
from ultralytics import YOLO

# --- 1. MODEL LOADING ---
print("Initializing official YOLOv8 model...")
# This will automatically download the official, large YOLOv8 model.
# This download is reliable and guaranteed to work.
model = YOLO('yolov8l.pt')  # 'l' stands for large, a powerful variant
print("✅ Model loaded successfully!")

# --- 2. DEFINE TARGET OBJECTS ---
# You can add more object names to this list from the 80 available classes.
# Examples: 'knife', 'baseball bat', 'scissors', etc.
VANDALISM_OBJECTS = ['spray can']

# --- 3. INITIALIZE VIDEO CAPTURE ---
video_capture = cv2.VideoCapture(0)  # Use 0 for your default webcam

# --- 4. THE MAIN LOOP ---
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # --- 5. GET PREDICTIONS ---
    results = model(frame, verbose=False)

    # --- 6. PROCESS & DISPLAY RESULTS ---
    for box in results[0].boxes:
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Check if the detected object is in our target list and has high confidence
        if class_name in VANDALISM_OBJECTS and confidence > 0.6:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw a red box around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Create and display the label
            label = f"VANDALISM RISK: {class_name.upper()} ({confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 🚨 This is where you would trigger your hardware action

    # Display the resulting frame
    cv2.imshow('S.A.H.A.R.A. - Vandalism Module (Press "q" to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. CLEANUP ---
video_capture.release()
cv2.destroyAllWindows()
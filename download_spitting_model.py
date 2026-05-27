import cv2
from roboflow import Roboflow

# --- 1. FILL IN YOUR FINAL DETAILS ---

# 🛑 PASTE YOUR NEWEST, SECRET API KEY HERE
ROBOFLOW_API_KEY = "3EnjAELcduturMzqpye3"

# --- These details are from your project ---
ROBOFLOW_WORKSPACE_ID = "ecs-project-3noyw"
ROBOFLOW_PROJECT_ID = "spit-detection-nlkqi-egijm"
MODEL_VERSION = 2


# --- 2. CONNECT TO YOUR MODEL (ROBUST METHOD) ---
print("🚀 Connecting to your model on Roboflow's servers...")
try:
    # Initialize Roboflow with your API key
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    # Get your specific workspace, project, and version
    workspace = rf.workspace(ROBOFLOW_WORKSPACE_ID)
    project = workspace.project(ROBOFLOW_PROJECT_ID)
    version = project.version(MODEL_VERSION)
    
    # This is the model object we'll use for predictions
    model = version.model
    
    print("✅ Connection successful. Starting detector...")

except Exception as e:
    print(f"❌ Could not connect to Roboflow. Please check your API Key and other IDs.")
    print(e)
    exit()


# --- 3. INITIALIZE VIDEO CAPTURE ---
video_capture = cv2.VideoCapture(0)

# --- 4. THE MAIN DETECTION LOOP ---
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # --- 5. SEND FRAME TO ROBOFLOW AND GET PREDICTIONS ---
    results = model.predict(frame, confidence=50, overlap=50).json()

    # --- 6. PROCESS AND DISPLAY RESULTS ---
    for prediction in results['predictions']:
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        x1, y1 = int(x - width / 2), int(y - height / 2)
        x2, y2 = int(x + width / 2), int(y + height / 2)

        # Draw the box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Create and display the label
        label = f"{prediction['class']}: {prediction['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('S.A.H.A.R.A. - Spitting Detector (Press "q" to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. CLEANUP ---
video_capture.release()
cv2.destroyAllWindows()
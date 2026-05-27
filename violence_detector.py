import cv2
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from collections import deque

# --- 1. CONFIGURATION & MODEL LOADING ---
print("Initializing model and processor...")
# This will automatically download the model from Hugging Face the first time
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
print("Model loaded successfully!")

# Constants for the model
SEQUENCE_LENGTH = 16  # The model processes 16 frames at a time
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

# --- 2. INITIALIZE VIDEO CAPTURE ---
video_capture = cv2.VideoCapture(0)  # Use 0 for your default webcam

# --- 3. THE MAIN LOOP ---
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # --- 4. PREPROCESS & PREDICT ---
    # The model expects RGB images, but OpenCV provides BGR, so we convert the color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames_queue.append(rgb_frame)

    display_label = "Status: Collecting Frames..."

    # We only predict once we have a full sequence of 16 frames
    if len(frames_queue) == SEQUENCE_LENGTH:
        # Prepare the video frames for the model
        inputs = processor(list(frames_queue), return_tensors="pt")

        # Get the model's prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted class ID and the label name
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        # --- 5. DECISION LOGIC ---
        # A list of keywords that indicate violence. You can customize this.
        violence_keywords = ["fighting", "punching", "kicking", "wrestling", "shooting", "sword fighting", "assault"]

        # Check if any keyword is in the model's prediction
        if any(keyword in predicted_label for keyword in violence_keywords):
            display_label = f"VIOLENCE DETECTED: {predicted_label}"
            # This is where you will add your hardware trigger later
        else:
            display_label = f"NonViolence: {predicted_label}"

    # --- 6. DISPLAY THE RESULTS ---
    cv2.putText(frame, display_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('S.A.H.A.R.A. - Violence Module (Press "q" to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. CLEANUP ---
video_capture.release()
cv2.destroyAllWindows()
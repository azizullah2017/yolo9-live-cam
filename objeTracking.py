import cv2
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov9c.pt')  # You can use 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', etc.

# Check if GPU is available and use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Reduce the resolution of the input frames
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Process every nth frame
frame_skip = 2
frame_count = 0

# Main loop to process the video stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to increase processing speed
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Apply the YOLO model
    results = model(frame)

    # Draw the bounding boxes on the frame
    for result in results[0].boxes:
        # Extract coordinates and convert them to integers
        x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().detach().numpy())
        
        # Extract confidence and class id
        confidence = float(result.conf.cpu().detach().numpy())
        class_id = int(result.cls.cpu().detach().numpy())
        label = f'{model.names[class_id]} {confidence:.2f}'
        
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw the label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Live', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

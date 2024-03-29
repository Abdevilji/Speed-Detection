import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sort.sort import *
import time
from math import dist
from util import get_car, read_license_plate, write_csv

# Check if OpenCV is built with CUDA support
print(cv2.getBuildInformation())

# Load YOLO models
vehicle_model = YOLO('./Models/yolov8s.pt')
license_plate_model = YOLO('./Models/license_plate_detector.pt')

# Initialize SORT tracker
tracker = Sort()

# Open the video capture
cap = cv2.VideoCapture('./Videos/test.mp4')

# Initialize variables
vehicles = [2, 3, 5, 7]
frame_number = -1
results = {}
car_speeds = {}  # Dictionary to store car speeds

# Speed detection parameters
cy1 = 273
cy2 = 368
offset = 6
vh_down = {}

# CUDA stream for asynchronous processing
stream = cv2.cuda_Stream()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    results[frame_number] = {}

    # Convert frame to GPU Mat
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame, stream=stream)

    # Detect vehicles
    detections = vehicle_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Convert detections to numpy array
    detections_np = np.array(detections_, dtype=np.float32)

    # Upload detections to GPU
    gpu_detections = cv2.cuda_GpuMat(detections_np.reshape(-1, 5).astype(np.float32))
    gpu_detections = gpu_detections.reshape(-1, 1, 5)

    # Track vehicles
    track_ids = tracker.update(gpu_detections, stream=stream)

    # Download tracked objects
    tracked_objects = track_ids.download(stream=stream)
    tracked_objects = tracked_objects.reshape(-1, 5)

    # Process each detected vehicle
    for bbox in tracked_objects:
        x3, y3, x4, y4, id = bbox
        x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)  # Convert to integers
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        # Calculate car speed
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time = time.time() - vh_down[id]
                if id not in car_speeds:
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    car_speeds[id] = a_speed_kh  # Store car speed in the dictionary

    # Detect license plates
    license_plates = license_plate_model(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, tracked_objects)

        if car_id != -1:
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            if license_plate_text is not None:
                results[frame_number][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                      'text': license_plate_text,
                                      'bbox_score': score,
                                      'text_score': license_plate_text_score}}

    # Display frame
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Video Simulation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write results
write_csv(results, './CSV/test2.csv')

# Write car IDs and speeds to another CSV file
car_speeds_df = pd.DataFrame(list(car_speeds.items()), columns=['Car_ID', 'Speed_Km_h'])
car_speeds_df.to_csv('./CSV/car_speeds.csv', index=False)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

# Load YOLO model
model = YOLO('./Models/yolov8s.pt')

# Open the text file to store car ID and speed
with open('./speed.txt', 'w') as file:
    # Callback function for mouse event (unused in this version)
    def mouse_callback(event, x, y, flags, param):
        pass

    # Open the video capture
    video_path = './Videos/test.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Read class names from coco.txt
    with open("./coco.txt", "r") as class_file:
        class_list = class_file.read().split("\n")

    # Initialize variables
    cy1 = int(0.54 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Line 1 position (54% of frame height)
    cy2 = int(0.736 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Line 2 position (73.6% of frame height)
    offset = 6
    vh_list = {}
    car_speeds = {}
    tracker = Tracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for processing
        frame = cv2.resize(frame, (1020, 500))

        # Performing object detection
        results = model.predict(frame)
        boxes = results.xyxy[0].cpu().numpy()

        # Extracting car bounding boxes
        list_car = []
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            if conf > 0.5 and class_list[int(class_id)] in ['car', 'motorcycle', 'truck']:
                list_car.append([int(x1), int(y1), int(x2), int(y2)])

        # Update tracker with detected car bounding boxes
        bbox_id_car = tracker.update(list_car)

        # Processing each detected car
        for bbox in bbox_id_car:
            x1, y1, x2, y2, id = bbox
            cy = (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Calculating car speed
            if cy1 - offset < cy < cy1 + offset:
                vh_list[id] = time.time()
            if id in vh_list and cy2 - offset < cy < cy2 + offset:
                elapsed_time = time.time() - vh_list[id]
                distance = abs(cy2 - cy1) / 100  # Distance between lines in meters
                if elapsed_time > 0:
                    speed_ms = distance / elapsed_time
                    speed_kmh = speed_ms * 3.6
                    car_speeds[id] = speed_kmh  # Storing car speed in the dictionary

                    # Write car ID and speed to the file
                    file.write(f"Car ID: {id}, Speed: {speed_kmh} Km/h\n")

                    # Displaying car ID and speed on the frame
                    cx = (x1 + x2) // 2
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"{int(speed_kmh)} Km/h", (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)

        # Drawing lines and text on the frame
        cv2.line(frame, (0, cy1), (frame.shape[1], cy1), (255, 255, 255), 1)
        cv2.putText(frame, 'Line 1', (10, cy1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
        cv2.line(frame, (0, cy2), (frame.shape[1], cy2), (255, 255, 255), 1)
        cv2.putText(frame, 'Line 2', (10, cy2 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)

        # Displaying the frame
        cv2.imshow("Vehicle Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Close the file after the loop
    file.close()

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

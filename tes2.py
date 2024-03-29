import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time
import csv

# Function to perform vehicle speed detection
def perform_speed_detection(video_path, model_path, output_csv_path):
    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Load YOLO model
    model = YOLO(model_path)

    # Open the CSV file to store car ID and speed
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Car ID', 'Speed (Km/h)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Initialize variables
        tracker = Tracker()
        cy1, cy2 = 273, 368
        offset = 6
        vh_down = {}
        counter = {}
        car_speeds = {}

        while True:    
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 500))

            # Performing object detection
            # Performing object detection
            results = model.predict(frame)
            list_car = []
# print(results)
            for result in results:
    # Access the boxes attribute for each result
                boxes = result.boxes.xyxy[0].cpu().numpy().tolist()
                print("Boxes:", boxes)  # Print out the boxes object for examination
                x1, y1, x2, y2 = boxes  # Accessing elements of the box list directly
                # Assuming class_id is 2 for cars
                list_car.append([int(x1), int(y1), int(x2), int(y2), 2])

            # list_car = []

            # # Extracting car bounding boxes
            # for box in boxes:
            #     x1, y1, x2, y2 = box  # Accessing elements of the box list directly
            #     # Assuming class_id is 2 for cars
            #     list_car.append([int(x1), int(y1), int(x2), int(y2), 2])

            # Updating tracker with detected car bounding boxes
            bbox_id_car = tracker.update(list_car)

            # Processing each detected car
            for bbox in bbox_id_car:
                x1, y1, x2, y2, car_id = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Calculating car speed
                if cy1 - offset < cy < cy1 + offset:
                    vh_down[car_id] = time.time()

                if car_id in vh_down and cy2 - offset < cy < cy2 + offset:
                    elapsed_time = time.time() - vh_down[car_id]
                    if car_id not in counter:
                        counter[car_id] = True
                        distance = 10  # meters
                        speed_ms = distance / elapsed_time
                        speed_kph = speed_ms * 3.6
                        car_speeds[car_id] = speed_kph

                        # Write car ID and speed to the CSV file
                        writer.writerow({'Car ID': car_id, 'Speed (Km/h)': speed_kph})

                        # Displaying car ID and speed on the frame
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, str(car_id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(frame, str(int(speed_kph)) + 'Km/h', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # Drawing lines and text on the frame
            cv2.line(frame, (319, cy1), (981, cy1), (255, 255, 255), 1)
            cv2.putText(frame, 'Line 1', (250, 270), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            cv2.line(frame, (268, cy2), (977, cy2), (255, 255, 255), 1)
            cv2.putText(frame, 'Line 2', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            car_count = len(counter)
            cv2.putText(frame, 'carcount:-' + str(car_count), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # Displaying the frame
            cv2.imshow("Vehicle Speed Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Entry point of the program
if __name__ == "__main__":
    video_path = 'D:/Project/Speed Detection/test.mp4'
    model_path = 'yolov8s.pt'
    output_csv_path = 'D:/Project/Speed Detection/speed.csv'
    perform_speed_detection(video_path, model_path, output_csv_path)

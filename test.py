import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time
from math import dist
from sort.sort import *
from util import get_car, read_license_plate, write_csv


# Open the video capture


# Load YOLO model
model = YOLO('./Models/yolov8s.pt')
license_plate_detector = YOLO('./Models/license_plate_detector.pt')
# Open the text file to store car ID and speed
with open('./speed.txt', 'w') as file:
    def RGB(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE :  
            colorsBGR = [x, y]
            print(colorsBGR)
            

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    cap=cv2.VideoCapture('./Videos/test.mp4')


    my_file = open("./coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n") 

    # Initialize variables
    count = 0
    cy1 = 273
    cy2 = 368
    offset = 6
    vh_down = {}
    counter = []
    car_speeds = {}
    tracker=Tracker()

    while True:    
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))

        # Performing object detection
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        list_car = []

        # Extracting car bounding boxes
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'car'  in c :
                list_car.append([x1, y1, x2, y2])
        bbox_id_car = tracker.update(list_car)

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            # print(license_plate)
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, bbox_id_car)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                # print(license_plate_crop)

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 127, 255, cv2.THRESH_BINARY)

                license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                if license_plate_text is not None:
                    results[count][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

        # Processing each detected car
        for bbox in bbox_id_car:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
            
            # Calculating car speed
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                vh_down[id] = time.time()
            if id in vh_down:
                if cy2 < (cy + offset) and cy2 > (cy - offset):
                    elapsed_time = time.time() - vh_down[id]
                    if counter.count(id) == 0:
                        counter.append(id)
                        distance = 10 # meters
                        a_speed_ms = distance / elapsed_time
                        a_speed_kh = a_speed_ms * 3.6
                        car_speeds[id] = a_speed_kh  # Storing car speed in the dictionary

                        # Write car ID and speed to the file
                        file.write(f"Car ID: {id}, Speed: {a_speed_kh} Km/h\n")

                        # Displaying car ID and speed on the frame
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        
        # Drawing lines and text on the frame
        cv2.line(frame, (319, cy1), (981, cy1), (255, 255, 255), 1)
        cv2.putText(frame, 'Line 1', (250, 270), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        cv2.line(frame, (268, cy2), (977, cy2), (255, 255, 255), 1)
        cv2.putText(frame, 'Line 2', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        d = len(counter)
        cv2.putText(frame, 'carcount:-' + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Displaying the frame
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Close the file after the loop
    file.close()
write_csv(results, './CSV/test2.csv')
# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

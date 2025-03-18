import os
import cv2
from ultralytics import YOLO
from tracker import Tracker


video_path = "./people.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

model = YOLO("yolov8n.pt")

tracker = Tracker()

while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = r
            x1=int(x1)
            y1=int(y1)
            x2=int(x2)
            y2=int(y2)
            class_id=int(class_id)

            detections.append([x1, y1, x2, y2, score])


        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 0), 2)


    cv2.imshow("Frame", frame)
    cv2.waitKey(25)

    ret, frame = cap.read()


cap.release()
cv2.destroyAllWindows()


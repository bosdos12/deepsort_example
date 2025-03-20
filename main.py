import os
import cv2
from ultralytics import YOLO
from tracker import Tracker
import cvzone

class DeepSortTest:
    def __init__(self):
        self.video_path = "./people.mp4"
        self.video_out_path = "./people_out.mp4"

        self.cap = cv2.VideoCapture(self.video_path)

        self.results = []

        # cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

        self.model = YOLO("yolov8n.pt")

        self.tracker = Tracker()


        self.run_inferrence()

    def run_inferrence(self):
        while True:

            # cap_out.write(frame)
            ret, self.frame = self.cap.read()
            self.results = self.model(self.frame)

            for result in self.results:
                detections = []
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    class_id = int(class_id)

                    detections.append([x1, y1, x2, y2, score])

                self.tracker.update(self.frame, detections)

                for track in self.tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    track_id = track.track_id

                    cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 0), 2)

                    # Add label
                    cv2.putText(self.frame,
                                f"{str(track_id)}",
                                (int(x1), int(y1)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, #scale
                                (0, 255, 0),
                                2, #thickness
                                cv2.LINE_AA)

            cv2.imshow("Frame", self.frame)
            cv2.waitKey(1)



# cap.release()
# cap_out.release()
# cv2.destroyAllWindows()



if __name__ == "__main__":
    DeepSortTest()
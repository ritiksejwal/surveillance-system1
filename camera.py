import cv2
from imutils.video import VideoStream
import imutils
import time
import numpy as np

class VideoCamera(object):
    def __init__(self, flip=False):
        self.vs = VideoStream(src=0).start()
        self.flip = flip
        time.sleep(2.0)
        self.object_classifier = cv2.CascadeClassifier("models/facial_recognition_model.xml")
        print("Classifier loaded:", not self.object_classifier.empty())

    def __del__(self):
        self.vs.stop()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.rot90(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_object(self):
        found_objects = False
        frame = self.flip_if_needed(self.vs.read()).copy()

        if frame is None:
            print("Empty frame")
            return None, found_objects

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        objects = self.object_classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(objects) > 0:
            found_objects = True

        # Draw a rectangle around the objects
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return ret, jpeg.tobytes(), found_objects

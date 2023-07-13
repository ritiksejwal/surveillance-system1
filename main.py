import cv2
import sys
from flask import Flask, render_template, Response
from flask_basicauth import BasicAuth
import time
import threading
from mail import sendEmail

email_update_interval = 600  
object_classifier = cv2.CascadeClassifier("models/facial_recognition_model.xml")  

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_object(self, classifier):
        success, frame = self.video.read()
        if not success:
            return None, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if len(objects) > 0:
            return frame, True
        else:
            return frame, False

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'usrid'
app.config['BASIC_AUTH_PASSWORD'] = 'pass'
app.config['BASIC_AUTH_FORCE'] = True

basic_auth = BasicAuth(app)
last_epoch = 0
video_camera = None

def check_for_objects():
    global video_camera, last_epoch
    while True:
        try:
            if video_camera is not None:
                frame, found_obj = video_camera.get_object(object_classifier)
                if found_obj and (time.time() - last_epoch) > email_update_interval:
                    last_epoch = time.time()
                    print("Sending email...")
                    sendEmail(frame)
                    print("done!")
        except Exception as e:
            print("Error sending email:", str(e))

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    video_camera = VideoCamera()

    t = threading.Thread(target=check_for_objects, args=())
    t.daemon = True
    t.start()

    app.run(host='0.0.0.0', debug=False)

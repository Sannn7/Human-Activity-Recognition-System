import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from playsound import playsound
from threading import Thread
import tensorflow as tf 
import time

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)



classes = ['Closed', 'Open']
face_cascade = cv2.CascadeClassifier("model/data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("model/data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("model/data/haarcascade_righteye_2splits.xml")
model = load_model("model/drowiness_new7.h5")

alarm_on = False
alarm_sound = "model/data/alarm.mp3"
status1 = ''
status2 = ''






class Camera():
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.start_time = time.time() 
        self.stop_time = self.start_time + 20
        
    def __del__(self):
        self.video.release()
        
    def get_feed(self):
        stat, frame = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
        
        is_decoded = (time.time() >= self.stop_time) # stop stream after 3 seconds
        
        return jpeg.tobytes(), is_decoded
    
# ---

def get_camera():
    return Camera()
    
def gen(camera):
    while True:
        frame, is_decoded = camera.get_feed() 
        if is_decoded:
            print('stop stream')
            # insert code here
             
            break
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'




def detect_drowsiness(): 
    count = 0
    camera = cv2.VideoCapture(0) 
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            height = frame.shape[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                left_eye = left_eye_cascade.detectMultiScale(roi_gray)
                right_eye = right_eye_cascade.detectMultiScale(roi_gray)
                for (x1, y1, w1, h1) in left_eye:
                    cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                    eye1 = roi_color[y1:y1+h1, x1:x1+w1]
                    eye1 = cv2.resize(eye1, (145, 145))
                    eye1 = eye1.astype('float') / 255.0
                    eye1 = img_to_array(eye1)
                    eye1 = np.expand_dims(eye1, axis=0)
                    pred1 = model.predict(eye1)
                    status1=np.argmax(pred1)
                    #print(status1)
                    #status1 = classes[pred1.argmax(axis=-1)[0]]
                    break

                for (x2, y2, w2, h2) in right_eye:
                    cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
                    eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
                    eye2 = cv2.resize(eye2, (145, 145))
                    eye2 = eye2.astype('float') / 255.0
                    eye2 = img_to_array(eye2)
                    eye2 = np.expand_dims(eye2, axis=0)
                    pred2 = model.predict(eye2)
                    status2=np.argmax(pred2)
                    #print(status2)
                    #status2 = classes[pred2.argmax(axis=-1)[0]]
                    break

                # If the eyes are closed, start counting
                if status1 == 2 and status2 == 2:
                #if pred1 == 2 and pred2 == 2:
                    count += 1
                    cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    # if eyes are closed for 10 consecutive frames, start the alarm
                    if count >= 10:
                        cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        if not alarm_on:
                            alarm_on = True
                            # play the alarm sound in a new thread
                            # t = Thread(target=start_alarm, args=(alarm_sound,))
                            t = Thread(target=playsound('model/data/alarm.mp3'))
                            t.daemon = True
                            t.start()
                else:
                    cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    count = 0
                    alarm_on = False

            # cv2.imshow("Drowsiness Detector", frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   

# def start_alarm(sound):
#     """Play the alarm sound"""
#     playsound('data/alarm.mp3')





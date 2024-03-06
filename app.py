from flask import Flask , render_template , Response
from utils import detect_drowsiness
from Fight_utils import start_streaming , loadModel
import cv2
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fight')
def fight():
   return render_template('fight_detection_live.html')


@app.route('/fight_detect2')
def fight_detect2():
    model =  loadModel("model/fight_detection_model.pth")
    return Response(start_streaming(model= model , streamingPath= 0), mimetype='multipart/x-mixed-replace; boundary=frame')

#replace detect_drowsiness()

@app.route('/sleepy_driver')
def sleepy_driver():
    return render_template('sleepy_driver_live.html')


@app.route('/sleepy_driver2')
def sleepy_driver2():
    return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)

# ffmpeg -i static/assets/img/example_fight2.mp4 -vcodec copy -acodec copy -ss 00:00:00 -t 00:03:32 static/assets/img/example_fight.mp4



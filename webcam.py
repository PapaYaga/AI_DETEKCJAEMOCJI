import time
import threading
import json
import sys

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, Response, render_template, url_for

from Network import Recognition

SIZE_FACE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

result_from_nn = None
image_for_nn = None

face_cascade = cv2.CascadeClassifier("./haarcascade_files/haarcascade_frontalface_default.xml")

response = {
    "face": None,
    "emotion1": -1,
    "emotion2": -1
}

network = Recognition()
network.build_network()
network.load_model()


def process_image(image):
    data_image = image # Image.fromarray(image).convert('RGB')
    image = np.array(data_image)[:, :, ::-1].copy()

    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC)

    background = np.zeros((150, 150), np.uint8)
    background[:, :] = 200
    background[int(((150 / 2) - (SIZE_FACE / 2))):int(((150 / 2) + (SIZE_FACE / 2))),
    int(((150 / 2) - (SIZE_FACE / 2))):int(((150 / 2) + (SIZE_FACE / 2)))] = image
    image = background

    cascade_classifier = cv2.CascadeClassifier('./haarcascade_files/haarcascade_frontalface_default.xml')
    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3)

    if not len(faces) > 0:
        return None
    biggest_face = faces[0]

    for face in faces:
        if face[2] * face[3] > biggest_face[2] * biggest_face[3]:
            biggest_face = face

    face = biggest_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        return None

    return image

size = []
for i in range(10):
    size.append(0)

lastResponse = "error"

def getResponse(frame, width, height):
    global result_from_nn, image_for_nn, response
    response = {
        "face": False,
        "emotion1": None,
        "emotion2": None
    }

    emotion1_index = None
    emotion2_index = None

    if frame is None:
        return response

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')
    faces = face_cascade.detectMultiScale(gray, 1.1, minSize=(100, 100))
    
    max_size = 0
    i = 0
    yy = 0
    xx = 0
    ww = 0
    hh = 0
    
    max_width = width 
    max_height = height

    for (x, y, w, h) in faces:
        i += 1
        
        size[i] = int(w*h)
        if(size[i] > max_size):
            max_size = size[i]
            multiplier = 0
            if round(h*multiplier) < y and round(h*multiplier) + y + h < max_height:
                y = y - round(h*multiplier)
                h = h + round(h*multiplier)*2
            if round(w*multiplier) < x and round(w*multiplier) + x + w < max_width:
                x = x - round(w*multiplier)
                w = w + round(w*multiplier)*2
            yy = int(y)
            xx = int(x)
            ww = int(w)
            hh = int(h)

    
    if(i > 0):
        image = frame[yy+2:yy+hh,xx+2:xx+ww]
        image_for_nn = process_image(image)

        result = network.predict(image_for_nn) 


        if result is not None:
            emotion_max = 0.0
            emotion_max2 = 0.0
            for index2, emotion in enumerate(EMOTIONS):
                if result[0][index2] > emotion_max:
                    emotion_max2 = emotion_max
                    emotion2_index = emotion1_index

                    emotion_max = result[0][index2] 
                    emotion1_index = index2
                    
                else:
                    if result[0][index2] > emotion_max2:
                        emotion_max2 = result[0][index2] 
                        emotion2_index = index2    

            if emotion_max2 < 0.2 or emotion_max2 < emotion_max * 0.5:
                emotion2_index = None
            
            
        else:
            emotion1_index = None
            emotion2_index = None

        response = {
            "face": True,
            "emotion1": emotion1_index,
            "emotion2": emotion2_index
        }

        print("results are: ",emotion1_index, emotion2_index, file=sys.stderr)
        if emotion1_index != None:
            mood = EMOTIONS[emotion1_index]
            with open("mood.txt", "w") as moodFile:
                moodFile.write(mood)
            


    else:
        response = {
            "face": False,
            "emotion1": -1,
            "emotion2": -1
        }

    return response
                    
 
app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return Response(open('./templates/getImage.html').read(), mimetype="text/html")


@app.route('/sendImage', methods=['POST'])
def image():
    received_image = request.files['image']  
    image = Image.open(received_image)
    img = np.array(image)  
    width, height = image.size
    response = getResponse(img, width, height)
    response_json = json.dumps(response)

    return Response(response_json)

@app.route('/getMood', methods=['GET'])
def mood():
    mood = "error"
    with open("mood.txt", "r") as moodFile:
        mood = moodFile.read()
        print("mood api called: ", mood, file=sys.stderr)
    
    response = {
            "mood": mood
        }

    response_json = json.dumps(response)

    return Response(response_json) 

if __name__ == '__main__':
    app.run()
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from Network import Recognition
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

SIZE_FACE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

correct_angry = 0
all_angry = 0
correct_disgusted = 0
all_disgusted = 0
correct_fearful = 0
all_fearful = 0
correct_happy = 0
all_happy = 0
correct_sad = 0
all_sad = 0
correct_surprised = 0
all_surprised = 0
correct_neutral = 0
all_neutral = 0


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

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


def emotion_to_vec(index):
    vector = np.zeros(len(EMOTIONS))
    vector[index] = 1.0
    return vector


def data_to_image(data):
    data_image = np.fromstring(str(data), dtype=np.uint8, sep=' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()
    data_image = format_image(data_image)
    return data_image


data = pd.read_csv('fer2013.csv')
labels = []
images = []
total = data.shape[0]

network = Recognition()
network.build_network()
network.load_model()

emotion_found = 0

results = (7,7)
results = np.zeros(results)

for index, row in data.iterrows():
    emo = row['emotion']
    image = data_to_image(row['pixels'])

    if image is not None:
        if emo == 0:
            all_angry += 1
        elif emo == 1:
            all_disgusted += 1
        elif emo == 2:
            all_fearful += 1
        elif emo == 3:
            all_happy += 1
        elif emo == 4:
            all_sad += 1
        elif emo == 5:
            all_surprised += 1
        elif emo == 6:
            all_neutral += 1

        result = network.predict(image)

        if result is not None:
            emotion_max = 0.0
            index_max = -1
            for index2, emotion in enumerate(EMOTIONS):
                if result[0][index2] > emotion_max:
                    index_max = index2
                    emotion_max = result[0][index2]
                    emotion_second = emotion_found
                    emotion_found = emotion
                #print(emotion, ': ', result[0][index2])

            if index_max == 0 and emo == 0:
                correct_angry += 1
            elif index_max == 1 and emo == 1:
                correct_disgusted += 1
            elif index_max == 2 and emo == 2:
                correct_fearful += 1
            elif index_max == 3 and emo == 3:
                correct_happy += 1
            elif index_max == 4 and emo == 4:
                correct_sad += 1
            elif index_max == 5 and emo == 5:
                correct_surprised += 1
            elif index_max == 6 and emo == 6:
                correct_neutral += 1

            results[index_max][emo] += 1
            

        else:
            emotion_found = None
            index_max = None

        print(index)
        #print("All: " + str(all_angry)+str(all_disgusted)+str(all_fearful)+str(all_happy)+str(all_sad)+str(all_surprised)+str(all_neutral))
        #print(np.array(results))
        #print(np.round(results/results.sum(0)[None,:]*100,2))

print("Angry: " + str(correct_angry / all_angry) + " %" + "Number of tries: " + str(all_angry))
print("Disgusted: " + str(correct_disgusted / all_disgusted) + " %" + "Number of tries: " + str(all_disgusted))
print("Fearful: " + str(correct_fearful / all_fearful) + " %" + "Number of tries: " + str(all_fearful))
print("Happy: " + str(correct_happy / all_happy) + " %" + "Number of tries: " + str(all_happy))
print("Sad: " + str(correct_sad / all_sad) + " %" + "Number of tries: " + str(all_sad))
print("Surprised: " + str(correct_surprised / all_surprised) + " %" + "Number of tries: " + str(all_surprised))
print("Neutral: " + str(correct_neutral / all_neutral) + " %" + "Number of tries: " + str(all_neutral))
print(np.round(results,0))
print(np.round(results/results.sum(0)[None,:]*100,2))
print(np.round(results/results.sum(1)[:,None]*100,2))
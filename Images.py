import cv2
import pandas as pd
import numpy as np
from PIL import Image

SIZE_FACE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def format_image(image):
    try:
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        background = np.zeros((150, 150), np.uint8)
        background[:, :] = 200
        background[int(((150 / 2) - (SIZE_FACE/2))):int(((150/2)+(SIZE_FACE/2))),
        int(((150/2)-(SIZE_FACE/2))):int(((150/2)+(SIZE_FACE/2)))] = image
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
index = 1
total = data.shape[0]

all = 0
notNone = 0
emo0 = 0
emo1 = 0
emo2 = 0
emo3 = 0
emo4 = 0
emo5 = 0
emo6 = 0
training = 0

for index, row in data.iterrows():
    emotion = emotion_to_vec(row['emotion'])
    image = data_to_image(row['pixels'])
    all += 1
    if image is not None and row['Usage'] == 'Training':
        notNone += 1
        if emotion[0] == 1:
            emo0 +=1
            for i in range(1):
                labels.append(emotion)
                images.append(image)
        elif emotion[1] == 1:
            emo1 +=1
            for i in range(1):
                labels.append(emotion)
                images.append(image)
        elif emotion[2] == 1:
            emo2 +=1
            for i in range(1):
                labels.append(emotion)
                images.append(image)
        elif emotion[3] == 1:
            emo3 +=1
            labels.append(emotion)
            images.append(image)
        elif emotion[4] == 1:
            emo4 +=1
            for i in range(1):
                labels.append(emotion)
                images.append(image)
        elif emotion[5] == 1:
            emo5 +=1
            for i in range(1):
                labels.append(emotion)
                images.append(image)
        elif emotion[6] == 1:
            emo6 +=1
            labels.append(emotion)
            images.append(image)

    if row['Usage'] == 'Training':
            training += 1
    index += 1
    print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

print("Total: " + str(len(labels)))
print(all)
print(index)
print(training)
print(notNone)
print(emo0)
print(emo1)
print(emo2)
print(emo3)
print(emo4)
print(emo5)
print(emo6)

np.save('./files/labels.npy', labels)
np.save('./files/images.npy', images)

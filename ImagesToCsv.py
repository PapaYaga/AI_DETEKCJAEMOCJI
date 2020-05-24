import os
from os.path import join

from PIL import Image
import numpy as np

SIZE_FACE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

additional_images_dir = join(".", "images")

with open("./finetuning.csv", 'w') as output_file:
    for image_filename in os.listdir(additional_images_dir):

        mood_name = image_filename.split("_")[0]
        mood = EMOTIONS.index(mood_name)

        image_path = join(additional_images_dir, image_filename)

        image = Image.open(image_path).resize((SIZE_FACE, SIZE_FACE)).convert("L")

        image_as_array = np.array(image, dtype=np.uint8)
        array_as_list = list(image_as_array.tostring())
        data_string = " ".join(str(num) for num in array_as_list)

        print(mood, data_string, "Training", sep=",", file=output_file)





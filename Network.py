import os
from os.path import join

import tflearn as tfl
import numpy as np
from sklearn.model_selection import train_test_split

SIZE_FACE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


class Recognition:
    def build_network(self):

        self.network = tfl.input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1])
        self.network = tfl.conv_2d(self.network, 64, 5, activation='relu')
        self.network = tfl.max_pool_2d(self.network, 3, strides=2)
        self.network = tfl.conv_2d(self.network, 64, 5, activation='relu')
        self.network = tfl.max_pool_2d(self.network, 3, strides=2)
        self.network = tfl.conv_2d(self.network, 128, 4, activation='relu')
        self.network = tfl.dropout(self.network, 0.5)
        self.network = tfl.fully_connected(self.network, 3072, activation='relu')
        self.network = tfl.fully_connected(
        self.network, len(EMOTIONS), activation='softmax')

        self.network = tfl.regression(
            self.network,
            optimizer='momentum',
            loss='categorical_crossentropy'
        )

        self.model = tfl.DNN(
            self.network,
            checkpoint_path='./files',
            max_checkpoints=1,
            tensorboard_verbose=2
        )

    def train_net(self):
        _images = np.load(join('./files', 'images.npy'))
        _images = _images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        _labels = np.load(join('./files', 'labels.npy'))
        _labels = _labels.reshape([-1, len(EMOTIONS)])

        self.images, self.images_test, self.labels, self.labels_test = train_test_split(
            _images,
            _labels,
            test_size=0.20,
            random_state=42
        )

        self.build_network()

        self.model.fit(
            self.images, self.labels,
            validation_set=(self.images_test,
                            self.labels_test),
            n_epoch=20,
            batch_size=50,
            shuffle=True,
            show_metric=True,
            snapshot_step=200,
            snapshot_epoch=True,
            run_id='emotion_recognition'
        )

        self.save_model()

    def save_model(self):
        self.model.save(join('./files', 'saved_model'))
        print('[+] Model trained and saved at ' + 'saved_model')

    def load_model(self):
        if os.path.isfile(join('./files', 'saved_model')):
            self.model.load(join('./files', 'saved_model'))
            print('[+] Model loaded from ' + 'saved_model')

    def predict(self, image):
        if image is None:
            return None
        image = image.reshape([1, SIZE_FACE, SIZE_FACE, -1])
        return self.model.predict(image)

if __name__ == "__main__":
    network = Recognition()
    network.train_net()
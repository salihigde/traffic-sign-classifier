import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.models import load_model
from PIL import Image
import os
from io import StringIO
import time
from glob import glob

class TLClassifier(object):
    def __init__(self):
        self.sign_classes = ['Red', 'Green', 'Yellow']

        self.model = load_model('saved_models/model.h5')
        self.graph = tf.get_default_graph()

    def get_classification(self, image, i):
        img_copy = np.copy(image)

        img_resize = cv2.resize(img_copy, (32, 32))
        cv2.imwrite('sample_imgs/test'+ str(i) + '.jpg', img_resize)

        img_resize = np.expand_dims(img_resize, axis=0).astype('float32')

        img_resize = (img_resize / 255.)

        with self.graph.as_default():
            predict = self.model.predict(img_resize, batch_size=1, verbose=1)

            print predict

            tl_color = self.sign_classes[np.argmax(predict)]

            print tl_color

if __name__ == '__main__':
        tl_cls = TLClassifier()
        TEST_IMAGE_PATHS = glob(os.path.join('test_imgs/', '*.jpg'))
        i=0
        for image_path in TEST_IMAGE_PATHS:
            img = cv2.imread(image_path)
            img_np = np.asarray(img, dtype="uint8")
            print('Processing following file:', image_path)
            start = time.time()
            tl_cls.get_classification(img_np, i)
            i += 1
            end = time.time()
            print('Classification time: ', end-start)

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

        self.model = load_model('saved_models/model_final_epochs_25_batch_32.h5')
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        img_copy = np.copy(image)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

        img_resize = cv2.resize(img_copy, (32, 32))
        img_resize = np.expand_dims(img_resize, axis=0).astype('float32')

        #cv2.imwrite('sample_imgs/test.jpg', img_resize)

        img_resize = (img_resize / 255.) - 0.5

        with self.graph.as_default():
            predict = self.model.predict(img_resize, batch_size=1, verbose=1)

            print predict

            tl_color = self.sign_classes[np.argmax(predict)]

            print tl_color

if __name__ == '__main__':
        tl_cls = TLClassifier()
        TEST_IMAGE_PATHS = glob(os.path.join('test_imgs/', '*.jpg'))
        for image_path in TEST_IMAGE_PATHS:
            img = Image.open(image_path)
            img_np = np.asarray(img, dtype="uint8" )
            img_np_copy = np.copy(img_np)
            print('Processing following file:', image_path)
            start = time.time()
            tl_cls.get_classification(img_np_copy)
            end = time.time()
            print('Classification time: ', end-start)

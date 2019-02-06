import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
import glob

import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

import keras
from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Activation, Flatten

def build_model():
    num_classes = 3
    model = Sequential()

    model.add(Conv2D(filters=8, kernel_size=5, strides=(2, 2), input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=5, strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=3, strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))

    model.summary()

    return model

def train_model(data, data_valid, model, epochs):
    X_train, y_train = data[0], data[1]

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    start_time = time.time()

    history = model.fit(X_train, y_train,
                        shuffle=True,
                        nb_epoch=epochs,
                        validation_data=(data_valid[0], data_valid[1])).history

    end_time = time.time()

    model.save('saved_models/model.h5')

    print('')
    print('Training time (seconds): ', end_time - start_time)

    return history, model

def evaluate(data_test, model, batch_size, visual=True):
    x_test = data_test[0]
    y_test = data_test[1]

    evaluation = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print('Model Accuracy = %.2f' % (evaluation[1]))

    if visual:
       predict = model.predict(x_test, batch_size=batch_size)

       fig, axs = plt.subplots(3, 3, figsize=(11, 8), facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.5, wspace=.001)
       axs = axs.ravel()

       for i in range(9):
          axs[i].imshow(cv2.cvtColor(x_test[i], cv2.COLOR_BGR2RGB))
          axs[i].set_title('Label: ' + str(np.argmax(y_test[i])) + ', Predict: '+str(np.argmax(predict[i])))

def get_training_images():
    #base_image_path = 'training_images/'
    base_image_path = 'training_imgs/'
    light_colors = ["red", "green", "yellow"]
    data = []
    color_counts = np.zeros(3)
    for color in light_colors:
        for img_file in glob.glob(os.path.join(base_image_path, color, "*")):
            img = cv2.imread(img_file)
            if not img is None:
                img = cv2.resize(img, (32, 32))
                label = light_colors.index(color)
                data.append((img, label, img_file))
                color_counts[light_colors.index(color)] += 1

                img_bright = img_brightness(img)
                data.append((img_bright, label, img_file))
                color_counts[light_colors.index(color)] += 1

                img_flip = cv2.flip(img, 1)
                data.append((img_flip, label, img_file))
                color_counts[light_colors.index(color)] += 1
                
    return data

def create_model_directory():
    saved_model_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(saved_model_dir):
           os.makedirs(saved_model_dir)

    return saved_model_dir

def img_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = 0.25 + np.random.uniform()
    hsv[:,:,2] = hsv[:,:,2] * brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def get_transformed_data(X_train, X_test, X_valid, y_train, y_test, y_valid):
    encoder = LabelBinarizer()
    encoder.fit(y_train)
    X_train /= 255.
    X_valid /= 255.
    X_test /= 255.
    y_train_onehot = encoder.transform(y_train)
    y_test_onehot = encoder.transform(y_test)
    y_valid_onehot = encoder.transform(y_valid)
    data_train = [X_train, y_train_onehot]
    data_test = [X_test, y_test_onehot]
    data_valid = [X_valid, y_valid_onehot]

    return data_train, data_test, data_valid

def preprocess_data(data):
    random.shuffle(data)
    X, y = [], []
    for d in data:
        X.append(d[0])
        y.append(d[1])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.09, random_state=832275)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.093, random_state=832275)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')

    return X_train, X_test, X_valid, y_train, y_test, y_valid

if __name__ == '__main__':
    saved_model_dir = create_model_directory()

    data = get_training_images()

    X_train, X_test, X_valid, y_train, y_test, y_valid = preprocess_data(data)
    data_train, data_test, data_valid = get_transformed_data(X_train, X_test, X_valid, y_train, y_test, y_valid)

    batch_size = 32
    epochs = 25

    print("Number of training samples: %d, Number of validation samples: %d, Number of test samples: %d" % (len(X_train), len(X_valid), len(X_test)))

    model = build_model()

    plot_model(model, to_file='model.png')

    history, model = train_model(data_train, data_valid, model, epochs)

    plt.plot(history['loss'], linewidth=2.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    evaluate(data_test, model, batch_size)

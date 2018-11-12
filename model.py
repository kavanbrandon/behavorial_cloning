import csv
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
import sklearn

 # Constants
correction = 0.25
batch_size = 64

lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

def generator(samples, batch_size=64):
    n_samples = len(samples)
    random.shuffle(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    path = './IMG/' + filename
                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)

                measurement = float(batch_sample[3])
                # measurement from center image
                measurements.append(measurement)
                # measurement from left image
                measurements.append(measurement + correction)
                # measurement from right image
                measurements.append(measurement - correction)

            # Image Augmentation
            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

 # Split test and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

 # Run generator formulas
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Normalize and crop the images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Nvidia model architecture
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_samples) * 5, nb_epoch=5, validation_data=validation_generator, nb_val_samples = len(validation_samples) * 5, verbose=1)
model.save('model.h5')

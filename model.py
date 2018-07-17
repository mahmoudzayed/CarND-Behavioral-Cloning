import csv
import cv2
import numpy as np
import tensorflow as tf
import sklearn
import os
from random import shuffle
from sklearn.model_selection import train_test_split

samples = []

print('Loading the data')

with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
		

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

images = []
angles = []
"""
ch, row, col = 3, 80, 320  # Trimmed image format

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = 'data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    measurment = float(batch_sample[3])
                    images.append(image)
                    if i == 0 :
                        angles.append(measurment)
                        image_flipped = cv2.flip(cv2.imread(name),1)
                        images.append(image_flipped)
                        angles.append(measurment*-1.0)
                    if i == 1 :
                        angles.append(measurment+0.2)
                    if i == 2 :
                        angles.append(measurment-0.2)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
	
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=1000)
validation_generator = generator(validation_samples, batch_size=1000)
"""
	
for batch_sample in samples[1:]:
    for i in range(3):
        name = 'data/IMG/'+batch_sample[i].split('/')[-1]
        image = cv2.imread(name)
        measurment = float(batch_sample[3])
        images.append(image)
        if i == 0 :
            angles.append(measurment)
            image_flipped = cv2.flip(cv2.imread(name),1)
            images.append(image_flipped)
            angles.append(measurment*-1.0)
        if i == 1 :
            angles.append(measurment+0.2)
        if i == 2 :
            angles.append(measurment-0.2)
X_train = np.array(images)
y_train = np.array(angles)
print(np.shape(images[1]))

print('Creating the model')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))

model.compile(loss= 'mse', optimizer= 'adam')
model.fit(X_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True)
#model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')	

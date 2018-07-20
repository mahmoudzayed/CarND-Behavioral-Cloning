import csv
import cv2
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def data_import(Filename):
	# reading file content
    print('Reading file')
    samples = []
    with open(Filename) as csvfile:
	    reader = csv.reader(csvfile)
	    for line in reader:
		    samples.append(line)
    return samples 
	
	
def Extract_data(samples):

    print('Loading the data')

    images = []
    angles = []
	# looping for each line in the csv file
    for batch_sample in samples:
		# importing the images for each image and its flip
        for i in range(3):
		    #read the image
            name = 'data/IMG/'+batch_sample[i].split('/')[-1]
            image = cv2.imread(name)
            measurment = float(batch_sample[3])
            images.append(image)
			#import angle measurment for center image and its flip 
            if i == 0 :
                angles.append(measurment)
                image_flipped = cv2.flip(cv2.imread(name),1)
                images.append(image_flipped)
                angles.append(measurment*-1.0)
			#import angle measurment for left image
            if i == 1 :
                angles.append(measurment+0.2)
			#import angle measurment for right image
            if i == 2 :
                angles.append(measurment-0.2)
	# convert the array to be numpy array
    X_train = np.array(images)
    y_train = np.array(angles)
	
    return X_train, y_train
	
def model_design(X_train, y_train):
    print('Creating the model')
    
	# model design
	
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3)))
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
    model.add(Dense(1))
    # compiling the model
    model.compile(loss= 'mse', optimizer= 'adam')
    model.fit(X_train, y_train, nb_epoch=2, validation_split=0.2, shuffle=True)
	# save the model
    model.save('model.h5')	
	

# the file name that contain the data
Filename = 'data/driving_log.csv'
#import file content
samples = data_import(Filename)
#read images and measurments
X_train, y_train = Extract_data(samples)
#create and train the model
model_design(X_train, y_train)

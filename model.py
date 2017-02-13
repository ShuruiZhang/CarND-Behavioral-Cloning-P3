#!/usr/bin/python

#import libraries
import os
import numpy as np
#import matplotlib.pyplot as plt
import csv
from random import randint
#use train_test_split to divide data
from sklearn.model_selection import train_test_split

#import keras
from keras.layers import Input,Convolution2D, Maxpooling2D, Dropout
# from keras.layers import Dropout, Flatten, Lambda, AveragePooling2D
from keras.models import Sequential
# from keras.layers import *
# from keras.optimizers import Adam

import tensorflow as tf

import time


CAMERA_OFFSET=0.2
EPOCHS=5
BATCH_SIZE=120
#import data

log_path=os.getcwd()+'/data/driving_log.csv'
# img_path=os.getcwd()+'/data/IMG/*.jpg'

####read data#########
######################

from numpy import genfromtxt
def decode_filename(filename):
	return os.getcwd()+'/data/'+filename.decode("utf-8").strip()
files = genfromtxt(log_path,delimiter=',',dtype="|S50, |S50, |S50, float, float, float, float")
files = np.array(file)
src_img_names=[]
src_steering_angles=[]
for line in files:
	steering_angle = line[3]
	center= decode_filename(line[0])
	left= decode_filename(line[1])
	right= decode_filename(line[2])
	src_img_names.append(center)
	src_steering_angles.append(steering_angle)
	src_img_names.append(left)
	src_steering_angles.append(steering_angle+CAMERA_OFFSET)
	src_img_names.append(right)
	src_steering_angles.append(steering_angle-CAMERA_OFFSET)

#shuffle data
from sklearn.utils import shuffle
src_img_names,src_steering_angles= shuffle(src_img_names,src_steering_angles)
#reformat list as np array
src_img_names=np.array(src_img_names)
src_steering_angles=np.array(src_steering_angles)

print("data size:",src_img_names.shape[0])


def get_img_and_angle(index):
	img_path=src_img_names[index]
	steering_angle=src_steering_angles[index]
	img=scipy.ndimage.imread(img_path)
	random_flip = random.randint(0,1)
	if (random_flip==1):
		img = scipy.flipr(img)
		steering_angle= -steering_angle
	return img, steering_angle*100 

def normalize_grayscale(image_data):
	a = -0.5
	b=0.5
	grayscale_min = 0
	grayscale_max=255
	return a+(((image_data-grayscale_min)*(b-a)/(grayscale_max - grayscale_min)))

def generator():
	while True:
		features= []
		labels=[]
		weights=[]
		for index in range(len(src_img_names)):
			image, steering_angle = get_img_and_angle(index)
			image = normalize_grayscale(image)
			features.append(image)
			labels.append(steering_angle)
			weights.append(abs(steering_angle+0.1))

			if (len(features)>= BATCH_SIZE):
				x = np.array(features)
				y= np. array(labels)
				w= np.array(weights)

				features= []
				labels=[]
				weights=[]
				yield x, y, w





#######network architecture below#########
###as shown in previous labs
print("building network...")
from keras.layers import Dense, Flatten, Activation
model = Sequential()
#5 conv layers
model.add(Convolution2D(24,5,5),subsample=(2,2),input_shape=(80,160,3))
model.add(Activation('relu'))

model.add(Convolution2D(36,5,5),subsample=(2,2))
model.add(Activation('relu'))

model.add(Convolution2D(48,5,5),subsample=(2,2))
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3),subsample=(2,2))
model.add(Activation('softmax'))


model.add(Convolution2D(64,3,3),subsample=(2,2))
model.add(Activation('softmax'))

#4 fully connected layers
#add  dropout to avoid overfitting
model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
print('network was built.')


#train the model
print("training...")
model.compile('adam','mean_squared_error',['accuracy'])
print('model compiled.')

#save model
def save_model():
	print('saving model...')
	model.save('model.h5')
	print('model saved.')

generator= generator(BATCH_SIZE)

for i in range(EPOCHS):
	model.fit(generator,len(src_img_names), 1, verbose=0.2)
	##build validation pipline
	validation_features = []
	validation_labels = []
	for j in range(int(len(src_img_names)/100)):
		#get a random image in the image data source
		index = random.randint(0,len(src_img_names)-1)
		validation_image, validation_angle = get_img_and_angle(index)
		validation_image= normalize_grayscale(validation_image)
		validation_features.append(validation_image)
		validation_labels.append(validation_angle)
	X_validation = np.array(validation_features)
	y_validation = np.array(validation_labels)
	metrics = model.evaluate(X_validation,y_validation,verbose=2)
	print('validation metrics result:', metrics)
	save_model()
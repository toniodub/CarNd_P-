#! /usr/bin/env python3

#file containing the The script used to create and train the model.
import cv2
import base64
from PIL import Image                                                            
import numpy as np                                                                     
import matplotlib.pyplot as plt                                                  
import glob
import csv
import pandas as pd
import json
import gc
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras.models import model_from_json
from keras.optimizers import *
from sklearn.utils import shuffle
from io import BytesIO

INIT_MODEL=0

# function that sort the input based on the steering value
def steering_filtering(X, y, steering):
	index_out=np.where(abs(y)>=steering)
	y_out=y[index_out[0]]
	X_out=X[index_out[0]]
	return X_out, y_out

print('version 2.0')
gc.collect()
#load data center
imageFolderPath = 'data/IMG/'
imagePath = glob.glob(imageFolderPath+'center*.jpg') 

#load and crop the data to remove sky and car
X_data_center = (np.array( [np.array((Image.open(imagePath[i])).crop((0,60,320,135))) for i in range(len(imagePath))])).astype(np.float32)

#load output
with open('data/driving_log.csv') as csv_file:
	df = pd.read_csv(csv_file)
	steering_data_center = df.steering.values
	throttle_data = df.throttle.values
	brake_data = df.brake.values
	speed_data = df.speed.values

#load data left
imagePath = glob.glob(imageFolderPath+'left*.jpg') 
#load and crop the data to remove sky and car
X_data_left = (np.array( [np.array((Image.open(imagePath[i])).crop((0,60,320,135))) for i in range(len(imagePath))])).astype(np.float32)
steering_data_left=np.copy(steering_data_center)+0.25

#load data right
imagePath = glob.glob(imageFolderPath+'right*.jpg') 
X_data_right=(np.array( [np.array((Image.open(imagePath[i])).crop((0,60,320,135))) for i in range(len(imagePath))])).astype(np.float32)
steering_data_right=np.copy(steering_data_center)-0.25

shape_data=np.shape(X_data_center)
shape_out=np.shape(steering_data_center)

print('preprocessing data')
X_data_temp=np.array([],dtype=np.float32).reshape(shape_data[0]*3,shape_data[1],shape_data[2],shape_data[3])
X_data_temp=np.concatenate((X_data_center,X_data_right,X_data_left))

steering_data_temp=np.array([],dtype=np.float32).reshape(shape_out[0]*3)
steering_data_temp=np.concatenate((steering_data_center,steering_data_right,steering_data_left))

#flipping image
X_data_flip=np.zeros(np.shape(X_data_temp),dtype=np.float32)
for n in range(np.shape(X_data_temp)[0]):
	X_data_flip[n,:,:,0]=cv2.flip(X_data_temp[n,:,:,0],flipCode=1)
	X_data_flip[n,:,:,1]=cv2.flip(X_data_temp[n,:,:,1],flipCode=1)
	X_data_flip[n,:,:,2]=cv2.flip(X_data_temp[n,:,:,2],flipCode=1)

steering_data_flip=-np.copy(steering_data_temp)

X_data=np.array([],dtype=np.float32).reshape(shape_data[0]*6,shape_data[1],shape_data[2],shape_data[3])
X_data=(np.concatenate((X_data_temp,X_data_flip))).astype(np.uint8)

steering_data_temp=np.array([],dtype=np.float32).reshape(shape_out[0]*6)
steering_data=np.concatenate((steering_data_temp,steering_data_flip))

steering_data=steering_data*10
steering_data=steering_data*steering_data*steering_data

#shuffle data
X_data, steering_data = shuffle(X_data, steering_data) 

#print crop image
img = Image.fromarray(X_data[0])
img.show()

#create test data separated from validation
X_test=X_data[0:1000]
steering_test = steering_data[0:1000]
#throttle_test = throttle_data[0:1000]
#brake_test = brake_data[0:1000]
#speed_test = speed_data[0:1000]

X_val=X_data[1000:1500]
steering_val = steering_data[1000:1500]

X_train=X_data[1500:-1]
steering_train = steering_data[1500:-1]
#throttle_train = throttle_data[1000:-1]
#brake_train = brake[1000:-1]
#speed_train = speed[1000:-1]

#preprocessing
datagen = ImageDataGenerator(
        rotation_range=5,
        height_shift_range=0.2,
        rescale=1./255,
        fill_mode='nearest')


if INIT_MODEL==0:
# create model
	print('creating model')

	model = Sequential()
	#1x1 kernel 3 output
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', input_shape=(75, 320, 3)))

	#5x5 kernel 32 output
	model.add(Activation('relu'))

	#5x5 kernel 32 output
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid'))
	model.add(Activation('relu'))

	#5x5 kernel 32 output
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid'))
	model.add(Activation('relu'))

	#3x3 kernel 32 output
	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(1164))
	model.add(Activation('relu'))

	model.add(Dense(100))
	model.add(Activation('relu'))
	
	model.add(Dense(100))
	model.add(Activation('relu'))

	model.add(Dense(50))
	model.add(Activation('relu'))

	model.add(Dense(1))

	#compile model
	model.compile('adam', 'mse')
else:
	print('loading model')
	with open('model.json', 'r') as jfile:
       
        	model = model_from_json(json.loads(jfile.read()))

	model.compile("adam", "mse")
	model.load_weights('model.h5')


datagen.fit(X_train)

print('start training')
nb_epoch=5
batch_size=32

val_gen=datagen.flow(X_val, steering_val, batch_size=batch_size)

for steering_th in range(5,0,-5):
	print('steering threshold:')
	print(steering_th/100)
	[X_train_temp, steering_train_temp]=steering_filtering(X_train,steering_train,steering_th/100)
	datagen.fit(X_train_temp)
	train_gen=datagen.flow(X_train_temp, steering_train_temp, batch_size=batch_size)

	# fits the model on batches with real-time data augmentation:
	model.fit_generator(train_gen, samples_per_epoch=min(len(X_train_temp),20000), nb_epoch=nb_epoch)
	gc.collect()

print()

#print(history.history.keys())
# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

#evaluation model
test_score = model.evaluate(X_test, steering_test)
print('test score:')
print(test_score)

print('exporting model H5')
model.save('my_model_arch.h5')

print('exporting model json')
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)

print('exporting model weight H5')
model.save_weights('model.h5')




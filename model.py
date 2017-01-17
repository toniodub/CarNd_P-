#! /usr/bin/env python3

#file containing the The script used to create and train the model.

from PIL import Image                                                            
import numpy as np                                                                     
import matplotlib.pyplot as plt                                                  
import glob
import csv
import pandas as pd
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from sklearn.utils import shuffle

#load data
imageFolderPath = 'data/IMG/'
imagePath = glob.glob(imageFolderPath+'center*.jpg') 

#load and crop the data to remove sky and car
X_data = np.array( [np.array((Image.open(imagePath[i])).crop((0,60,320,135))) for i in range(len(imagePath))])


print('shape of input data is : ')
print(np.shape(X_data))

#load output
with open('data/driving_log.csv') as csv_file:
	df = pd.read_csv(csv_file)
	steering_data = df.steering
	throttle_data = df.throttle
	brake_data = df.brake
	speed_data = df.speed

#print crop image
img = Image.fromarray(X_data[0])
img.show()

print('preprocessing data')
#shuffle data
X_data, steering_data = shuffle(X_data, steering_data) 


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
        rotation_range=20,
        width_shift_range=0,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        fill_mode='nearest')


# create model
print('creating model')

model = Sequential()
#1x1 kernel 3 output
model.add(Convolution2D(3, 1, 1, input_shape=(75, 320, 3)))

#3x3 kernel 32 output
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1))
#model.add(Activation('softmax'))

#compile model
model.compile('adam', 'mean_squared_error', ['accuracy'])
datagen.fit(X_train)



print('start training')
nb_epoch=1
batch_size=128

train_gen=datagen.flow(X_train, steering_train, batch_size=batch_size)
val_gen=datagen.flow(X_val, steering_val, batch_size=batch_size)

# fits the model on batches with real-time data augmentation:
history=model.fit_generator(train_gen, samples_per_epoch=len(X_train), nb_epoch=nb_epoch,validation_data=val_gen, nb_val_samples=500)

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
model.save('my_model.h5')

print('exporting model json')
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)

print('exporting model weight H5')
model.save_weights('my_model_weights.h5')




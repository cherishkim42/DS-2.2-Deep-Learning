import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tflearn.data_utils as du
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_healthy = os.listdir('/ignore-data/chest_xray/train/NORMAL/')
train_sick = os.listdir('/ignore-data/chest_xray/train/PNEUMONIA/')

image_height = 150
image_width = 150
batch_size = 10
epochs = 5

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(image_height,image_width,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, shear_range=0.2, zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('train/', target_size=(image_width, image_height), batch_size=batch_size, class_mode='binary')

test_set = test_datagen.flow_from_directory('test/', target_size=(image_width, image_height), batch_size=batch_size, class_mode='binary')

val_set = test_datagen.flow_from_directory('val/', target_size=(image_width, image_height), batch_size=1, shuffle=False, class_mode='binary')

reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, cooldown=2, min_lr=0.00001, verbose=1)

callbacks = [reduce_learning_rate]

run_model = model.fit_generator(training_set, steps_per_epoch=5216//batch_size, epochs=epochs, validation_data=test_set, validation_steps=624//batch_size, callbacks=callbacks)
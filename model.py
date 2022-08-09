import os
import shutil
from glob import glob
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D

# import matplotlib.pyplot as plt
# import numpy as np
# from keras.utils.np_utils import to_categorical
# import random
# from keras.models import load_model

DATA_DIR = 'data'
TRAIN_DIR = os.path.join('data', 'train')
VAL_DIR = os.path.join('data', 'val')

if os.path.exists(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)
if os.path.exists(VAL_DIR):
    shutil.rmtree(VAL_DIR)
    
shutil.copytree(os.path.join(DATA_DIR, 'open'), os.path.join(TRAIN_DIR, 'open'))
shutil.copytree(os.path.join(DATA_DIR, 'closed'), os.path.join(TRAIN_DIR, 'closed'))

open_img = glob(TRAIN_DIR + '/open/*.jpg')
closed_img = glob(TRAIN_DIR + '/closed/*.jpg')

open_train, open_val = train_test_split(open_img, test_size=0.3)
closed_train, closed_val = train_test_split(closed_img, test_size=0.3)

os.makedirs(os.path.join(VAL_DIR, 'open'))
for file in open_val:
  os.rename(file, file.replace('train', 'val'))

os.makedirs(os.path.join(VAL_DIR, 'closed'))
for file in closed_val:
  os.rename(file, file.replace('train', 'val'))


BS = 32
TS = (24, 24)
gen = ImageDataGenerator(rescale=1. / 255)
train_batch = gen.flow_from_directory(TRAIN_DIR, batch_size=BS, shuffle=True, color_mode='grayscale', class_mode='categorical', target_size=TS)
valid_batch = gen.flow_from_directory(VAL_DIR, batch_size=BS, shuffle=True, color_mode='grayscale', class_mode='categorical', target_size=TS)
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)), # 32 convolution filters used each of size 3x3
    MaxPooling2D(pool_size=(1, 1)), # choose the best features via pooling
    Conv2D(32, (3, 3), activation='relu'), # 32 convolution filters used each of size 3x3
    MaxPooling2D(pool_size=(1, 1)), # choose the best features via pooling
    Conv2D(64, (3, 3), activation='relu'), # 64 convolution filters used each of size 3x3
    MaxPooling2D(pool_size=(1, 1)), # choose the best features via pooling
    Dropout(0.25),  # randomly turn neurons on and off to improve convergence
    Flatten(),  # flatten since too many dimensions, we only want a classification output
    Dense(128, activation='relu'),  # fully connected to get all relevant data
    Dropout(0.5),  # one more dropout for convergence' sake :)
    Dense(2, activation='softmax') # output a softmax to squash the matrix into output probabilities
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_batch, epochs=15, steps_per_epoch=SPE, validation_data=valid_batch, validation_steps=VS)

print("Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])

model.save('models/cnnModel.h5', overwrite=True)


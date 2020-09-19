#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import tensorflow as tf
from tensorflow import keras


train_folder='chest_xray/train'
val_folder='chest_xray/val'
test_folder='chest_xray/test'



labels = ['PNEUMONIA', 'NORMAL']
img_size = 200
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)




train=get_data(train_folder)
val=get_data(val_folder)
test=get_data(test_folder)



from sklearn.model_selection import train_test_split
X = []
y = []

for feature, label in train:
    X.append(feature)
    y.append(label)

for feature, label in test:
    X.append(feature)
    y.append(label)
    
for feature, label in val:
    X.append(feature)
    y.append(label)


X = np.array(X).reshape(-1, img_size, img_size, 1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=32)



X_train=X_train/255
X_test=X_test/255
X_val=X_val/255



from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen =ImageDataGenerator(
       featurewise_center=False, 
       samplewise_center=False,  
       featurewise_std_normalization=False,  
       samplewise_std_normalization=False,  
       zca_whitening=False,  
       rotation_range=90, 
       zoom_range = 0.1, 
       width_shift_range=0.1,  
       height_shift_range=0.1,  
       horizontal_flip=True,  
       vertical_flip=True)  

datagen.fit(X_train)



from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
model=tf.keras.Sequential()

model.add(Conv2D(6, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization(axis=1))

model.add(Conv2D(4, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization(axis=1))

model.add(Conv2D(4, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization(axis=1))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))



model.compile(optimizer='sgd', loss='mse')



model.summary()




history = model.fit(datagen.flow(X_train, y_train, batch_size=10),validation_data=(X_val, y_val), epochs=15)




model.evaluate(X_train, y_train)




pred = model.predict(X_train)





predictions = model.predict(X_test)




binary_predictions = []
threshold = 0.5
for i in predictions:
    if i >= threshold:
        binary_predictions.append(1)
    else:
        binary_predictions.append(0) 





binary_predictions=np.array(binary_predictions)




from sklearn.metrics import accuracy_score, confusion_matrix



print("Accuracy :",accuracy_score(binary_predictions,y_test))




model.save('pneumonia_detection5.h5')







# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 09:06:33 2018

@author: ahmthaydrornk

"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np

classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(16, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 64, activation = 'relu'))
classifier.add(Dense(output_dim = 32, activation = 'relu'))
classifier.add(Dense(output_dim = 16, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   #shear_range = 0.2,
                                   #zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   #shear_range = 0.2,
                                   #zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 100,
                         nb_epoch = 8)

test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1, steps = 32)

for i in range(0,int(32)):
    if pred[i] < 0.5:
        pred[i] = 0
    else:
        pred[i] = 1

test_labels = []

for i in range(0,int(32)):
    test_labels.extend(np.array(test_set[i][1]))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, pred)

print("Karmaşıklık Matrisi")
print (cm)




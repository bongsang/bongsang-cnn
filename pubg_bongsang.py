import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Hyperparameters
epochs = 10
batch = 32

kernel = (3, 3)
stride = (1, 1)
pool = (3, 3)

learning_rate = 0.0001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
drop_rate = 0.8

num_classes = 1

save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'trained_model.h5'


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Data Loading
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (224, 224),
                                                 batch_size = batch,
                                                 class_mode = 'binary')
testing_set = test_datagen.flow_from_directory('data/validation',
                                            target_size = (224, 224),
                                            batch_size = batch,
                                            class_mode = 'binary')

# Neural Networks
model = Sequential()

# Layer 1
model.add(Conv2D(256, kernel_size=kernel, strides=stride, input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool))
model.add(Dropout(drop_rate))

# Layer 2
model.add(Conv2D(256, kernel_size=kernel, strides=stride))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool))
model.add(Dropout(drop_rate))

# Layer 3
model.add(Conv2D(128, kernel_size=kernel, strides=stride))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool))
model.add(Dropout(drop_rate))

# Flattening
model.add(Flatten())

# Fully Connection
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

opt = keras.optimizers.Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=eps)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# Training
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)


# Evaluating
model.evaluate(x_test, y_test)
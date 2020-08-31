import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Activation,\
    Concatenate
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import load_model, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.metrics
from utils import get_data, save_data, load_data, test_partition_data, val_test_partition_data, load_from_npy, load_train, load_test
from model import get_callbacks, create_ResNet50_model, create_custom_model
from preprocess import partition_save_data
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data_generators
from matplotlib import pyplot as plt
from utils import *

batch_size = 32
image_size = 100


train_generator, val_generator, test_generator = get_data_generators(image_size, batch_size)

callbacks = get_callbacks()

class_weights = {0: 1.0,
                1: 1.0}

#opt = Adam()

model = create_custom_model(image_size)


model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=2,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    class_weight=class_weights,
                    verbose=2)


model.save("model_custom_" + str(image_size) + "px.h5")

plot_acc(history)
plot_loss(history)

steps = test_generator.n//test_generator.batch_size


y_pred = model.predict_generator(test_generator, steps=steps, verbose=1)
y_test = test_generator.classes
y_pred = np.array([np.argmax(x) for x in y_pred])[0:(steps*batch_size)-1]
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(y_test, y_pred))


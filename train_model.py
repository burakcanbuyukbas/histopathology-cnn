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
#from keras.applications.inception_v3 import preprocess_input



# Normalize data
# input_train = X_train / 255

acc_per_fold = []
loss_per_fold = []


batch_size = 8


train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        vertical_flip=True)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True)

#test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    directory="data/train",
    target_size=(100, 100),
    batch_size=32
)

val_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=(100, 100),
        batch_size=32
)

# test_generator = test_datagen.flow_from_directory(
#         'data/test',
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode='binary')

callbacks = get_callbacks()

class_weights = {0: 1.0,
                1: 1.0}

#opt = Adam()

model = create_custom_model()
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


history = model.fit(train_generator,
                    steps_per_epoch=int(222018/32),
                    epochs=30,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=int(27753/32),
                    class_weight=class_weights,
                    verbose=2)



model.save("model1.h5")



# X_test, Y_test = load_test()
# input_test = X_test / 255
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
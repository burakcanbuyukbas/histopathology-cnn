import numpy as np
import pandas as pd
import os
import keras
from utils import get_data, save_data, load_data, test_partition_data, val_test_partition_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Activation,\
    Concatenate
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import load_model, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input as ResNetPreprocess
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.metrics
from keras.applications.inception_v3 import preprocess_input


def create_ResNet50_model():
    model = ResNet(weights='imagenet', include_top=False,
                   input_shape=None, pooling='avg')
    x = model.output
    x = Dropout(0.5)(x)
    preds = Dense(2, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=preds)

    return model

def create_custom_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same', input_shape=(100, 100, 3)))
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                   verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, verbose=1)

    filepath = "checkpoints/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpointer = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

    return [reduce_lr, early_stopping, checkpointer]
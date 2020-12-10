import numpy as np
import pandas as pd
import os
import keras
from utils import get_data, save_data, load_data, test_partition_data, val_test_partition_data
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Activation,\
    Concatenate, Conv1D, MaxPooling1D
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import load_model, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input as ResNetPreprocess
from keras.applications.vgg16 import VGG16 as VGG, preprocess_input as VGGPreprocess
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.metrics


def create_ResNet50_model(dropout=0.5, image_size=100):
    resnet = ResNet(weights='imagenet', include_top=False,
                   input_shape=(image_size, image_size, 3), pooling='avg')
    x = resnet.output
    x = Dropout(dropout)(x)
    preds = Dense(2, activation='softmax')(x)
    model = Model(inputs=resnet.input, outputs=preds)
    for layer in resnet.layers:
        layer.trainable = False

    return model

def create_VGG16_model(dropout=0.5, image_size=100):
    vgg = VGG(weights='imagenet', include_top=False,
                   input_shape=(image_size, image_size, 3), pooling='avg')
    x = vgg.output
    x = Dropout(dropout)(x)
    preds = Dense(2, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=preds)
    for layer in vgg.layers:
        layer.trainable = False

    return model

def create_custom_model(image_size=100):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
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

def get_callbacks(checkpoint_folder = "checkpoints/"):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                   verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, verbose=1)

    checkpoint_path = checkpoint_folder + "checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpointer = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

    return [reduce_lr, early_stopping, checkpointer]

def create_custom_model1(image_size=100):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    return model


def create_custom_model2(image_size=100):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    return model


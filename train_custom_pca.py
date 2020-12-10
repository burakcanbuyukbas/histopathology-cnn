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
from model import *
from preprocess import partition_save_data
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data_generators, array_generators
from matplotlib import pyplot as plt
from utils import *

batch_size = 32
image_size = 100
epochs = 20

X_train, X_val, Y_train, Y_val = train_test_split(np.load('pca/X_train.npy'), np.load('pca/Y_train.npy'),
                                                  test_size=0.1, random_state=42)

X_train = X_train.reshape((X_train.shape[0], 100, 1))
X_val = X_val.reshape((X_val.shape[0], 100, 1))

# train_generator, val_generator, test_generator = array_generators(X_train=X_train, Y_train=Y_train,
#                                                                   X_val=X_val, Y_val=Y_val,
#                                                                   X_test=np.load('pca/X_test.npy'),
#                                                                   Y_test=np.load('pca/X_test.npy'),
#                                                                   batch_size=batch_size, model='resnet')

callbacks = get_callbacks()

class_weights = {0: 1.0,
                1: 0.5}

#opt = Adam()

model = create_custom_pca_model(image_size)


model.compile(loss='binary_crossentropy', optimizer=Adam(0.1), metrics=['accuracy'])


history = model.fit(X_train, Y_train,
                    batch_size=100,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(X_val, Y_val),
                    class_weight=class_weights,
                    verbose=1)


model.save("model_custom_pca.h5")
try:
    plot_acc(history)
    plot_loss(history)
finally:
    X_test = np.load('pca/X_test.npy')
    X_test = X_test.reshape((20000, 100, 1))
    Y_test = np.load('pca/Y_test.npy')

    y_pred = model.predict(X_test, batch_size=100, verbose=1)
    y_pred = np.array([np.round(x) for x in y_pred])
    print(confusion_matrix(Y_test, y_pred, labels=[0, 1]))
    print(classification_report(Y_test, y_pred))


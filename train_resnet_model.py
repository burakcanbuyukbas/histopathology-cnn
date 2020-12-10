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
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data_generators

batch_size = 32
image_size = 100


train_generator, val_generator, test_generator = get_data_generators(image_size, batch_size, model='resnet')

callbacks = get_callbacks()

class_weights = {0: 1.0,
                1: 1.0}

#opt = Adam()

model = create_ResNet50_model(image_size)


# print("Stage 1: Transfer Learning")
#
# model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
#
# history = model.fit(train_generator,
#                     steps_per_epoch=train_generator.samples // batch_size,
#                     epochs=3,
#                     callbacks=callbacks,
#                     validation_data=val_generator,
#                     validation_steps=val_generator.samples // batch_size,
#                     class_weight=class_weights,
#                     verbose=2)
#
# print("First stage done.")
model.load_weights("checkpoints/resnet/checkpoint-03-0.72.hdf5")
print("Model loaded.")

# try:
#     val_loss_history = history.history['val_loss']
#     val_acc_history = history.history['val_accuracy']
#     loss_history = history.history['loss']
#     acc_history = history.history['accuracy']
# except KeyError:
val_loss_history = []
val_acc_history = []
loss_history = []
acc_history = []



# Stage 2:
print("Stage 2: Fine-tuning")
for layer in model.layers:
    layer.trainable = True
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])


history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=3,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    class_weight=class_weights,
                    verbose=2)

print("Second stage done. Please be good.")

try:
    loss_history = np.append(loss_history, history.history['loss'])
    acc_history = np.append(acc_history, history.history['accuracy'])
    val_loss_history = np.append(val_loss_history, history.history['val_loss'])
    val_acc_history = np.append(val_acc_history, history.history['val_accuracy'])
except KeyError:
    pass


model.save("model_resnet_" + str(image_size) + "px.h5")
np.save("model/model_resnet_" + str(image_size) + "px_losshistory.npy", loss_history)
np.save("model/model_resnet_" + str(image_size) + "px_losshistory.npy", acc_history)

steps = test_generator.n//test_generator.batch_size


y_pred = model.predict_generator(test_generator, steps=steps, verbose=1)
y_test = test_generator.classes
y_pred = np.array([np.argmax(x) for x in y_pred])[0:(steps*batch_size)-1]
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(y_test, y_pred))


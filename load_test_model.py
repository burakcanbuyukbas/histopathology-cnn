import numpy as np
import pandas as pd
import os
from model import get_callbacks, create_ResNet50_model, create_custom_model, create_VGG16_model
from preprocess import partition_save_data
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data_generators

batch_size = 32
image_size = 100

# model = resnet, vgg or else(no preprocess function)
train_generator, val_generator, test_generator = get_data_generators(image_size, batch_size, model='resnet')

callbacks = get_callbacks(checkpoint_folder="checkpoints/vgg/")


# load model
# model = load_model('checkpoints/custom-old/checkpoint-10-0.87.hdf5')
model = load_model('checkpoints/resnet/checkpoint-01-0.89.hdf5')
#model = load_model('checkpoints/vgg/checkpoint-03-0.81.hdf5')

steps = test_generator.n//test_generator.batch_size


y_pred = model.predict_generator(test_generator, steps=steps, verbose=1)
y_test = test_generator.classes
y_pred = np.array([np.argmax(x) for x in y_pred])[0:(steps*batch_size)-1]
print(confusion_matrix(y_test[:22700], y_pred[:22700], labels=[0, 1]))
print(classification_report(y_test[:22700], y_pred[:22700]))
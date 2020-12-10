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

model = load_model('model1.h5')
print(model.summary())

# vgg
# model = load_model('checkpoints/checkpoint-01-0.72.hdf5')
#
# print(model.summary())
#
#
model = load_model('checkpoints/vgg/checkpoint-01-0.80.hdf5')
#
print(model.summary())
# --------------------------------------***-------------------------------------- #
# resnet
# model = load_model('checkpoints/checkpoint-01-0.89.hdf5')
#
# print(model.summary())
#
#
# model = load_model('checkpoints/resnet/checkpoint-01-0.74.hdf5')
#
# print(model.summary())



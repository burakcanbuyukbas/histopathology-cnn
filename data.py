import pandas as pd
from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input as ResnetPreprocess
from keras.applications.vgg16 import VGG16 as Vgg, preprocess_input as VggPreprocess
from keras_preprocessing.image import ImageDataGenerator
import numpy as np



def get_data_generators(image_size=100, batch_size=32, model='resnet'):
    if(model=='resnet'):
        preprocess_function = ResnetPreprocess
    elif(model=='vgg'):
        preprocess_function = VggPreprocess
    else:
        preprocess_function = None

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        horizontal_flip=True,
        vertical_flip=True)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        horizontal_flip=False,
        vertical_flip=False)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        horizontal_flip=False,
        vertical_flip=False)

    train_generator = train_datagen.flow_from_directory(
        directory="data/train",
        target_size=(image_size, image_size),
        class_mode='binary',
        batch_size=batch_size
    )

    val_generator = val_datagen.flow_from_directory(
        directory='data/val',
        target_size=(image_size, image_size),
        class_mode='binary',
        batch_size=batch_size
    )

    test_generator = test_datagen.flow_from_directory(
        directory="data/test",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    return train_generator, val_generator, test_generator



def array_generators(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size = 32, model=None):
    if(model=='resnet'):
        preprocess_function = ResnetPreprocess
    elif(model=='vgg'):
        preprocess_function = VggPreprocess
    else:
        preprocess_function = None

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        horizontal_flip=False,
        vertical_flip=False)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        horizontal_flip=False,
        vertical_flip=False)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        horizontal_flip=False,
        vertical_flip=False)


    train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)

    val_generator = val_datagen.flow(X_val, Y_val, batch_size=batch_size)

    test_generator = test_datagen.flow(X_test, Y_test, batch_size=batch_size)

    return train_generator, val_generator, test_generator

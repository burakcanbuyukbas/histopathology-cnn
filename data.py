import pandas as pd
from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input as ResnetPreprocess
from keras_preprocessing.image import ImageDataGenerator




def get_data_generators(image_size=100, batch_size=32, model='resnet'):
    if(model=='resnet'):
        preprocess_function = ResnetPreprocess
    else:
        preprocess_function = None

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        horizontal_flip=True,
        vertical_flip=True)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        horizontal_flip=True,
        vertical_flip=True)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        horizontal_flip=True,
        vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        directory="data/train",
        target_size=(image_size, image_size),
        class_mode='categorical',
        batch_size=batch_size
    )

    val_generator = val_datagen.flow_from_directory(
        directory='data/val',
        target_size=(image_size, image_size),
        class_mode='categorical',
        batch_size=batch_size
    )

    test_generator = test_datagen.flow_from_directory(
        directory="data/test",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    return train_generator, val_generator, test_generator
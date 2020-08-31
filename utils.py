import numpy as np
import os
import cv2
import pandas as pd
from glob import glob
import fnmatch
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt


def get_data():

    X = []
    Y = []
    count = 0


    path = r'D:\Users\Burak\histopathology'
    patches = glob(path + '/IDC_regular_ps50_idx5/**/*.png', recursive=True)
    for filename in patches[0:10]:
        print(filename)

    negatives = fnmatch.filter(patches, '*class0.png')
    positives = fnmatch.filter(patches, '*class1.png')
    print("Negatives: " + str(len(negatives)))
    print("Positives: " + str(len(positives)))

    print("Every day I'm shufflin...'")
    random.shuffle(patches)
    print("Data shuffled.")

    for img in patches:
        image = cv2.imread(img)
        if img in negatives:
            Y.append(0)
            X.append(cv2.resize(image, (50, 50), interpolation=cv2.INTER_CUBIC))
        elif img in positives:
            Y.append(1)
            X.append(cv2.resize(image, (50, 50), interpolation=cv2.INTER_CUBIC))

        if count % 1000 == 0:
            print(str(count) + " / " + str(len(patches)))
            
        count = count +1


    X = np.array(X)
    Y = np.array(Y)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    return X, Y

def save_data(X, Y, path=""):
    print("Saving data...")
    np.save(path + 'X.npy', X)
    np.save(path + 'Y.npy', Y)

def save_data_train_test(X_train, Y_train, X_test, Y_test):
    print("Saving data...")
    np.save('X_train.npy', X_train)
    np.save('Y_train.npy', Y_train)
    np.save('X_test.npy', X_test)
    np.save('Y_test.npy', Y_test)

def val_test_partition_data(X, Y, val_ratio, test_ratio):
    print("Partitioning data...")
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_ratio, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=test_ratio, random_state=42)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def test_partition_data(X, Y, test_ratio):
    print("Partitioning data...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=42)
    return X_train, X_test, Y_train, Y_test


def load_data(X_path = 'X.npy', Y_path = 'Y.npy'):
    print("Loading data...")
    X = np.load(X_path)
    Y = np.load(Y_path)

    return X, Y


def load_from_npy(image_size=50):
    print("Loading data...")
    X_Train = np.load('X_train.npy', mmap_mode='r')
    X_Test = np.load('X_test.npy', mmap_mode='r')
    Y_Train = np.load('Y_train.npy', mmap_mode='r')
    Y_Test = np.load('Y_test.npy', mmap_mode='r')
    print("Train Benign: " + str(np.count_nonzero(Y_Train == 0)))
    print("Train Malignant: " + str(np.count_nonzero(Y_Train == 1)))

    print("Test Benign: " + str(np.count_nonzero(Y_Test == 0)))
    print("Test Malignant: " + str(np.count_nonzero(Y_Test == 1)))

    return X_Train, Y_Train, X_Test, Y_Test

def load_train():
    print("Loading data...")
    X_Train = np.load('X_train.npy', mmap_mode='r')
    Y_Train = np.load('Y_train.npy', mmap_mode='r')
    print("Train Benign: " + str(np.count_nonzero(Y_Train == 0)))
    print("Train Malignant: " + str(np.count_nonzero(Y_Train == 1)))
    return X_Train, Y_Train


def load_test():
    print("Loading data...")
    X_Test = np.load('X_test.npy', mmap_mode='r')
    Y_Test = np.load('Y_test.npy', mmap_mode='r')

    print("Test Benign: " + str(np.count_nonzero(Y_Test == 0)))
    print("Test Malignant: " + str(np.count_nonzero(Y_Test == 1)))
    return X_Test, Y_Test


def divideToFolders():
    X_train, X_val, X_test = [], [], []
    Y_train, Y_val, Y_test = [], [], []
    id = 0
    path = r'D:\Users\Burak\histopathology'
    train_positive_path = "data/train/1/"
    train_negative_path = "data/train/0/"
    val_positive_path = "data/val/1/"
    val_negative_path = "data/val/0/"
    test_positive_path = "data/test/1/"
    test_negative_path = "data/test/0/"

    patches = glob(path + '/IDC_regular_ps50_idx5/**/*.png', recursive=True)
    total = len(patches)
    
    for filename in patches[0:10]:
        print(filename)


    negatives = fnmatch.filter(patches, '*class0.png')
    positives = fnmatch.filter(patches, '*class1.png')
    print("Negatives: " + str(len(negatives)))
    print("Positives: " + str(len(positives)))

    print("Every day I'm shufflin...'")
    random.shuffle(patches)
    print("Data shuffled.")

    for img in patches:
        full_image = cv2.imread(img)
        image = cv2.resize(full_image, (100, 100), interpolation=cv2.INTER_CUBIC)
        if(id%10 == 0):#val
            if img in negatives:
                Y_val.append(0)
                cv2.imwrite(val_negative_path + str(id) + '.png', image)
                X_val.append(str(id) + '.png')
                id = id+1
            elif img in positives:
                Y_val.append(1)
                cv2.imwrite(val_positive_path + str(id) + '.png', image)
                X_val.append(str(id) + '.png')
                id = id+1
        elif(id%11 == 0 and id%10!=0):#test
            if img in negatives:
                Y_test.append(0)
                cv2.imwrite(test_negative_path + str(id) + '.png', image)
                X_test.append(str(id) + '.png')
                id = id+1
            elif img in positives:
                Y_test.append(1)
                cv2.imwrite(test_positive_path + str(id) + '.png', image)
                X_test.append(str(id) + '.png')
                id = id+1
        else:#train
            if img in negatives:
                Y_train.append(0)
                cv2.imwrite(train_negative_path + str(id) + '.png', image)
                X_train.append(str(id) + '.png')
                id = id+1
            elif img in positives:
                Y_train.append(1)
                cv2.imwrite(train_positive_path + str(id) + '.png', image)
                X_train.append(str(id) + '.png')
                id = id+1
        if(id%1000 == 0):
            print(str(id) + "/" + str(total))
    print("Saving data...")
    np.save('arrays/X_train.npy', np.array(X_train))
    np.save('arrays/Y_train.npy', np.array(Y_train))
    np.save('arrays/X_val.npy', np.array(X_val))
    np.save('arrays/Y_val.npy', np.array(Y_val))
    np.save('arrays/X_test.npy', np.array(X_test))
    np.save('arrays/Y_test.npy', np.array(Y_test))
    print("Save complete.")
    print("X_train: " + str(np.array(X_train).shape))
    print("Y_train: " + str(np.array(Y_train).shape))
    print("X_val: " + str(np.array(X_val).shape))
    print("Y_val: " + str(np.array(Y_val).shape))
    print("X_test: " + str(np.array(X_test).shape))
    print("Y_test: " + str(np.array(Y_test).shape))


def plot_acc(model):
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
def plot_loss(model):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
#X, Y = get_data()
#save_data(X, Y)
#divideToFolders()


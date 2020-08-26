import numpy as np
import os
import cv2
import pandas as pd
from utils import get_data, save_data, load_data, test_partition_data, save_data_train_test


#X, Y = get_data()
#save_data(X, Y)
def partition_save_data():
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = test_partition_data(X, Y, 0.1)
    save_data_train_test(X_train, Y_train, X_test, Y_test)


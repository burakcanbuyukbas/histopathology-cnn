import os
from keras.preprocessing.image import ImageDataGenerator
import shutil
import numpy as np


image_size = 100

def augment_data():
    aug_dir = 'aug_dir'
    img_dir = os.path.join(aug_dir, 'img_dir')
    img_list = os.listdir('data/train/1')

    # copy images from the class all images directory to the image directory
    for fname in img_list:
        # source path to image
        src = os.path.join('data\\train\\1', fname)
        # destination path to image
        dst = os.path.join(img_dir, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    #64401
    # point to a dir containing the images and not to the images themselves
    path = 'aug_dir'
    save_path = 'data/train/1'

    data_generator = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.125,
        height_shift_range=0.125,
        zoom_range=[1.2, 1.325],
        fill_mode='reflect')

    batch_size = 25

    aug_datagen = data_generator.flow_from_directory(path,
                                              save_to_dir=save_path,
                                              save_format='png',
                                              target_size=(image_size, image_size),
                                              batch_size=batch_size)

    # generate the augmented images and add them to the folder of all images

    num_aug_images_wanted = len(os.listdir('data/train/0')) # total number of images wanted in each class

    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted-num_files)/batch_size))

    for i in range(0, num_batches):
      next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree('aug_dir')
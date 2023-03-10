# link ref
# https://www.kaggle.com/code/mushfirat/denoise-images-using-autoencoders
# https://www.kaggle.com/code/bunnyyy/medical-image-denoising-using-autoencoders/notebook

# For ML Models

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img
import cv2

# For Data Processing
import numpy as np

# For Data Visualization
import matplotlib.pyplot as plt

# Miscellaneous
import os
import random

# Turn off warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# TRAIN_SIZE = 384
# INFERENCE_SIZE = 224
TRAIN_SIZE=1280
INFERENCE_SIZE=1280


# img = cv2.imread('./data/31366_00013.png')
# mean_denoising = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
# median_blur = cv2.medianBlur(img, 11)
# gaussian_blur = cv2.GaussianBlur(img, (3, 3), 0)
# bilateral = cv2.bilateralFilter(img, -1, 10, 5)

# cv2.imwrite('./data/result/mean_denoising.png', mean_denoising)
# cv2.imwrite('./data/result/median_blur.png', median_blur)
# cv2.imwrite('./data/result/gaussian_blur.png', gaussian_blur)
# cv2.imwrite('./data/result/bilateral.png', bilateral)


def open_images(paths, size=TRAIN_SIZE):
    '''
    Given an array of paths to images, this function opens those images,
    and returns them as an array of shape (None, Height, Width, Channels)
    '''
    images = []
    
    for path in paths:
        image = load_img(path, target_size=(size, size, 3))
        image = np.array(image)/255.0 # Normalize image pixel values to be between 0 and 1
        images.append(image)
    return np.array(images)

def add_noise(images, amount=0.1):
    '''
    Given an array of images [a shape of (None, Height, Width, Channels)],
    this function adds gaussian noise to every channel of the images
    '''
    # Create a matrix with values with a mean of 0 and standard deviation of "amount"
    noise = np.random.normal(0, amount, images.shape[0] * 
                             images.shape[1] * 
                             images.shape[2] *
                             images.shape[3]).reshape(images.shape)
    
    noise_img = images + noise
    return noise_img

def datagen(noise_deleted_path, ori_path, size=TRAIN_SIZE, batch_size=5):
    '''
    
    Given an array of images to noise_deleted_path,
    this function return batch of images as (noise_image, real_image)
    '''
    for x in range(0, len(noise_deleted_path), batch_size):
        batch_paths = noise_deleted_path[x:x+batch_size]
        batch_images = open_images(batch_paths, size=size)
        
        # amount = random.uniform(0, 0.2)
        # noise_images = add_noise(batch_images, amount=amount)
        noise_images = ori_path[x:x+batch_size]
        noise_images = open_images(noise_images, size=size)
        yield noise_images, batch_images
        
def plot_results(noise_image, reconstructed_image, image):
    w = 15
    h = len(noise_image)*3
    fig = plt.figure(figsize=(w, h))
    columns = 3
    rows = len(noise_image)
    for i in range(1, rows*columns, columns):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.title('Image with noise')
        plt.imshow(noise_images[int((i-1) / columns)])
        
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.title('Reconsturcted Image')
        plt.imshow(reconstructed_image[int((i-1) / columns)])
        
        fig.add_subplot(rows, columns, i+2)
        plt.axis('off')
        plt.title('Original Image')
        plt.imshow(image[int((i-1) / columns)])
        
    plt.savefig('reconstructed_result.png')
    
if __name__ == '__main__':
    
    # main_dir = '/home/Juneer_deeplearning_cookbook/data/Flickr_dataset/'
    main_dir = '/home/Juneer_deeplearning_cookbook/data/las_for_autoencoder/noise_deleted/'
    noise_dir = '/home/Juneer_deeplearning_cookbook/data/las_for_autoencoder/original/'
    # os.listdir(path)는 path 디렉토리 내의 모든 파일, 디렉토리를 list 형태로 return
    all_image_paths = [main_dir + file for file in os.listdir(main_dir) if file.endswith('.png')]
    
    print(f"Total number of images: {len(all_image_paths)}")
    train_image_paths = all_image_paths[:200]
    test_image_paths = all_image_paths[200:]
    noise_image_paths = [noise_dir + file for file in os.listdir(noise_dir) if file.endswith('.png')]
    
    
    # image = open_images([train_image_paths[27]])
    # noise_img = add_noise(image, amount=0.2)
    
    # fig = plt.figure(figsize=(10, 5))
    # # plot image
    # fig.add_subplot(1, 2, 1)
    # plt.axis('off')
    # plt.title('Image')
    # plt.imshow(image[0])
    # # plot image with noise
    # fig.add_subplot(1, 2, 2)
    # plt.axis('off')
    # plt.title('Image with noise')
    # plt.imshow(noise_img[0])
    
    # plt.show()
    # plt.savefig('sample.png')
    
    # image = Input(shape=(None, None, 3))      # Input()은 Keras Tensor 초기화를 위해 쓰이는 객체
    
    # # Encoder
    # l1 = Conv2D(64, (3, 3), padding='same', activation='relu',
    #             activity_regularizer=regularizers.l1(10e-10))(image)
    # l2 = Conv2D(64, (3, 3), padding='same', activation='relu',
    #             activity_regularizer=regularizers.l1(10e-10))(l1)
    
    # l3 = MaxPooling2D(padding='same')(l2)
    # l3 = Dropout(0.3)(l3)
    
    # l4 = Conv2D(128, (3,3), padding='same', activation='relu',
    #         activity_regularizer=regularizers.l1(10e-10))(l3)
    # l5 = Conv2D(128, (3,3), padding='same', activation='relu',
    #             activity_regularizer=regularizers.l1(10e-10))(l4)

    # l6 = MaxPooling2D(padding='same')(l5)
    # l7 = Conv2D(256, (3,3), padding='same', activation='relu',
    #             activity_regularizer=regularizers.l1(10e-10))(l6)
    
    # # Decoder
    # l8 = UpSampling2D()(l7)
    # l9 = Conv2D(128, (3, 3), padding='same', activation='relu',
    #             activity_regularizer=regularizers.l1(10e-10))(l8)
    # l10 = Conv2D(128, (3,3), padding='same', activation='relu',
    #         activity_regularizer=regularizers.l1(10e-10))(l9)
    
    # l11 = add([l5,l10])
    # l12 = UpSampling2D()(l11)
    # l13 = Conv2D(64, (3,3), padding='same', activation='relu',
    #             activity_regularizer=regularizers.l1(10e-10))(l12)
    # l14 = Conv2D(64, (3,3), padding='same', activation='relu',
    #             activity_regularizer=regularizers.l1(10e-10))(l13)

    # l15 = add([l14,l2])

    # decoded = Conv2D(3, (3,3), padding='same', activation='relu',
    #                 activity_regularizer=regularizers.l1(10e-10))(l15)
    # model = Model(image, decoded)
    
    # print(model.summary())
    
    # model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy')
    
    # batch_size=2
    # steps=int(len(train_image_paths)/batch_size)
    # epochs=50
    # for epoch in range(epochs):
    #     model.fit(datagen(train_image_paths, noise_image_paths, size=TRAIN_SIZE, 
    #                       batch_size=batch_size), epochs=1, steps_per_epoch=steps)
        
    # model.save('./saved_result')
    
    # batch_size=10
    # steps = int(len(test_image_paths)/batch_size)
    model = load_model("saved_result")
    # model.evaluate(datagen(test_image_paths, size=INFERENCE_SIZE, batch_size=batch_size), steps=steps)
    
    batch_size = 1

    # paths = random.sample(test_image_paths, batch_size)
    # paths = random.sample(noise_image_paths, batch_size)
    # paths = '/home/Juneer_deeplearning_cookbook/data/31366_00081.png'
    # images = open_images(paths, size=INFERENCE_SIZE)
    # # Amount of noise = random value between 0.1 and 0.15
    # amount = random.uniform(0.1,0.15)
    # noise_images = add_noise(images, amount=amount)
    
    # tp_paths = [noise_dir + path.split('/')[-1] for path in paths]
    # noise_images = open_images(tp_paths, size=INFERENCE_SIZE)
    tp_path = '/home/Juneer_deeplearning_cookbook/data/31370_01055.png'
    tp_file_name = tp_path.split('/')[-1]
    noise_images = open_images([tp_path], size=INFERENCE_SIZE)
    reconstructed = model.predict(noise_images)
    # print(reconstructed.dtype)
    # print(reconstructed.shape)
    # plt.imshow(reconstructed[0])
    # plt.savefig('fig1.png')
    # ori_result = cv2.normalize(images[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    result = cv2.normalize(reconstructed[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # for x in range((result.shape[0])):
    #     for y in range((result.shape[1])):
    #         for z in range((result.shape[2])):
    #             # print(result[x][y].shape)
    #             # exit()
    #             result[x][y][z] = np.max(result[x][y])
    # # reconstructed = reconstructed.astype(np.uint8)
    # reconstructed[0] = cv2.cvtColor(reconstructed[0], cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./original.png', images[0])
    # print(paths[0])
    cv2.imwrite('./reconstructed_' + tp_file_name + '.png', result)

    
    # plot_results(noise_images, reconstructed, images)
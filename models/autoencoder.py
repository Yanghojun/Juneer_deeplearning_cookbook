# pytorch
from torch import nn
import torch

# keras
from tensorflow import keras
from tensorflow.keras.optimizers import *
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))        # 상위 폴더인 Juneer_deeplearning_cookbook을 append
from base.keras_model_base import BaseModel                                         # 그 이유는 autoencoder.py 파일 입장에서 상위 디렉토리인
                                                                                    # base directory 를 import 하기 위함
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import *
from keras import Model, regularizers
import time
from tqdm import tqdm
from glob import glob
from tensorflow.keras import optimizers

class AutoEncoder(BaseModel):
    def __init__(self, config, train_dataloader, test_dataloader):
        super().__init__(config, train_dataloader, test_dataloader)       # 부모 클래스의 인스턴스 property 초기화 및
                                                    # keras 모델, 데이터로더 등의 초기화를 위한 초기 코드 흐름(함수 호출)을 실행한다.
        return

    def define_model(self):         # 부모 클래스의 define_model을 오버라이딩함
        if self.config.config_namespace.PYTORCH:
            model = Pytorch_AutoEncoder()

        else:
            image = Input(shape=(None, None, self.config.config_namespace.CHANNELS))      # Input()은 Keras Tensor 초기화를 위해 쓰이는 객체
        
            # Encoder
            l1 = Conv2D(64, (3, 3), padding='same', activation='relu',
                        activity_regularizer=regularizers.l1(10e-10))(image)
            l2 = Conv2D(64, (3, 3), padding='same', activation='relu',
                        activity_regularizer=regularizers.l1(10e-10))(l1)
            
            l3 = MaxPooling2D(padding='same')(l2)
            l3 = Dropout(0.3)(l3)
            
            l4 = Conv2D(128, (3,3), padding='same', activation='relu',
                    activity_regularizer=regularizers.l1(10e-10))(l3)
            l5 = Conv2D(128, (3,3), padding='same', activation='relu',
                        activity_regularizer=regularizers.l1(10e-10))(l4)

            l6 = MaxPooling2D(padding='same')(l5)
            l7 = Conv2D(256, (3,3), padding='same', activation='relu',
                        activity_regularizer=regularizers.l1(10e-10))(l6)
            
            # Decoder
            l8 = UpSampling2D()(l7)
            l9 = Conv2D(128, (3, 3), padding='same', activation='relu',
                        activity_regularizer=regularizers.l1(10e-10))(l8)
            l10 = Conv2D(128, (3,3), padding='same', activation='relu',
                    activity_regularizer=regularizers.l1(10e-10))(l9)
            
            l11 = add([l5,l10])
            l12 = UpSampling2D()(l11)
            l13 = Conv2D(64, (3,3), padding='same', activation='relu',
                        activity_regularizer=regularizers.l1(10e-10))(l12)
            l14 = Conv2D(64, (3,3), padding='same', activation='relu',
                        activity_regularizer=regularizers.l1(10e-10))(l13)

            l15 = add([l14,l2])

            # decoded = Conv2D(3, (3,3), padding='same', activation='relu',
            #                 activity_regularizer=regularizers.l1(10e-10))(l15)

            decoded = Conv2D(self.config.config_namespace.CHANNELS, (3,3), padding='same', activation='relu',
                            activity_regularizer=regularizers.l1(10e-10))(l15)
            model = Model(image, decoded)

        return model

    # def compile_model(self):
    #     # self.model.compile(optimizer=self.config.config_namespace.OPTIMIZER,
    #     #                     loss=self.config.config_namespace.LOSS)
    #     self.model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy')

    def fit_model(self):
        if self.config.config_namespace.PYTORCH:
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            
            # train
            for epoch in range(self.config.config_namespace.EPOCHS):
                running_loss = 0.0

                for idx, (x, y) in enumerate(tqdm(self.train_dataloader, ascii=True)):
                    output = self.model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    running_loss += loss.item()

                print(f"Epoch: {epoch + 1}, loss: {running_loss}")

        else:
            self.model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy')
            start_time = time.time()
            print("Training phase under progress, trained ConvNet model will be saved at path", self.config.config_namespace.SAVED_MODEL_DIR, " ...\n")

            self.model.fit_generator(
                self.train_dataloader,
                epochs=self.config.config_namespace.EPOCHS,
                validation_data=self.test_dataloader,
                workers=6,
                shuffle=True
            )

            end_time = start_time - time.time()

            self.train_time = end_time - start_time
            print("The model took %0.3f seconds to train. \n" %self.train_time)
            self.model.save(self.config.config_namespace.SAVED_MODEL_DIR)

    def open_images(self, paths, size=1280):
        '''
        Given an array of paths to images, this function opens those images,
        and returns them as an array of shape (None, Height, Width, Channels)
        '''
        images = []
        for path in paths:
            if self.config.config_namespace.CHANNELS == 1:
                image = load_img(path, color_mode='grayscale')
            elif self.config.config_namespace.CHANNELS == 3:
                image = load_img(path, target_size=(size, size, 3))    

            else:
                print("Error! Channels must be 1 or 3")
                exit()

            image = np.array(image)/255.0 # Normalize image pixel values to be between 0 and 1
            images.append(image)
        return np.array(images)

    def predict(self):
        model = keras.models.load_model(self.config.config_namespace.LOAD_MODEL_DIR)
        print(f"Load model from {self.config.config_namespace.LOAD_MODEL_DIR}")

        print("Is this printed..?")
        
        batch_size = 1
    
        # tp_path = '/home/Juneer_deeplearning_cookbook/dataset/31370_01055.png'
        tp_path = '/home/Juneer_deeplearning_cookbook/dataset/las_data_annotated_v3_23_03_03/test/31366_00082.png'
        tp_file_name = tp_path.split('/')[-1]
        noise_images = self.open_images([tp_path], size=1280)
        reconstructed = model.predict(noise_images)
        
        # result = cv2.normalize(reconstructed[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        # result = cv2.normalize(reconstructed, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        print('./reconstructed_' + tp_file_name)
        cv2.imwrite('./reconstructed_' + tp_file_name, (reconstructed[0] * 255).astype(np.uint8))

    def generate_data(self):
        model = keras.models.load_model(self.config.config_namespace.LOAD_MODEL_DIR)
        print(f"Load model from {self.config.config_namespace.LOAD_MODEL_DIR}")
        os.makedirs(self.config.config_namespace.GENERATED_DATA_PATH, exist_ok=True)
        data_paths = glob(self.config.config_namespace.TRAIN_DATA_PATH + '*.png')
        data_paths.extend(glob(self.config.config_namespace.TEST_DATA_PATH + '*.png'))

        for path in tqdm(data_paths, desc="Generating Data..."):
            file_name = path.split('/')[-1]
            input_data = self.open_images([path], size=1280)
            output_data = model.predict(input_data)
            output_data = cv2.normalize(output_data[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(self.config.config_namespace.GENERATED_DATA_PATH + file_name, output_data)

class Pytorch_AutoEncoder(nn.Module):
    def __init__(self):
        super(Pytorch_AutoEncoder, self).__init__()

        # Encoder
        self.cnn_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.cnn_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        # Decoder
        self.tran_l1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )
        
        self.tran_l2 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn_l1(x)
        x = self.cnn_l2(x)
        x = self.tran_l1(x)
        x = self.tran_l2(x)

        return x

if __name__ == '__main__':
    model = AutoEncoder(dataset=None)
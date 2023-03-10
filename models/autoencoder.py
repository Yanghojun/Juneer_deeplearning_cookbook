import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))        # 상위 폴더인 Juneer_deeplearning_cookbook을 append
from base.keras_model_base import BaseModel                                         # 그 이유는 autoencoder.py 파일 입장에서 상위 디렉토리인
                                                                                    # base directory 를 import 하기 위함
from keras.models import Sequential
from keras.layers import *
from keras import Model, regularizers

class AutoEncoder(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)       # 부모 클래스의 인스턴스 property 초기화 및 
                                        # keras 모델, 데이터로더 등의 초기화를 위한 초기 코드 흐름(함수 호출)을 실행한다.
        return
    
    def define_model(self):         # 부모 클래스의 define_model을 오버라이딩함
        image = Input(shape=(None, None, 3))      # Input()은 Keras Tensor 초기화를 위해 쓰이는 객체
    
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

        decoded = Conv2D(3, (3,3), padding='same', activation='relu',
                        activity_regularizer=regularizers.l1(10e-10))(l15)
        model = Model(image, decoded)

        return model

if __name__ == '__main__':
    model = AutoEncoder(dataset=None)
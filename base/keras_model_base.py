import numpy as np
import os
from keras.models import Sequential
from keras.callbacks import History, ModelCheckpoint

class BaseModel(object):
    def __init__(self, config, dataset):
        self.config = config
        self.model = self.define_model()        # define_model 함수는 자식 클래스에서 재정의된다. 
                                                # 그래서 model.summary() 출력이 가능하다.
        print(self.model.summary())
        self.dataset = dataset
        
    def define_model(self):
        raise NotImplementedError
        # print("this is basemodel define_model")
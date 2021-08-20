'''
BeatGan preprocess function
'''

import os
import numpy as np
from numpy.lib.function_base import delete
from sklearn.utils import shuffle
import torch
from  torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import json
import glob
import math
import pandas as pd
from typing import Union
import gzip
import pickle
from utils.log import Logger

np.random.seed(42)

def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1

def load_data(opt):
    # for train data (normal)
    normal_path = sorted(glob.glob(os.path.join(opt.dataroot,"normal", "**","*.json"), recursive=True), key=os.path.getmtime)    # 수정한 날짜 순으로 하면 내가 원하는 파일 순서로 저장됨
    
    air_normal_data = []
    air_normal_data_code = []
    air_normal_file_name = []
    
    for i in normal_path:
        with open(i) as f:
            json_data = json.load(f)
            file_name = str(json_data['area']) +"_"+ json_data['start_date']
            json_data = json_data['data']
        
        element_name = opt.dataroot.split('/')[2]

        element = json_data[element_name]
        element_code = json_data[element_name + '_CODE']
        data = [element]
        data_code = [element_code]

        data = np.array(data)
        data_code = np.array(data_code)
        air_normal_data.append(data)
        air_normal_data_code.append(data_code)
        air_normal_file_name.append(file_name)

    # for test data (abnormal)
    abnormal_path = sorted(glob.glob(os.path.join(opt.dataroot,"abnormal", "*.json")), key=os.path.getmtime)
    
    air_abnormal_data = []
    air_abnormal_data_code = []
    air_abnormal_file_name = []
    
    for i in abnormal_path:
        with open(i) as f:
            json_data = json.load(f)
            file_name = str(json_data['area']) +"_"+ json_data['start_date']
            json_data = json_data['data']

        element_name = opt.dataroot.split('/')[2]
        # print("element name of abnormal: " + str(element_name))

        element = json_data[element_name]
        element_code = json_data[element_name+'_CODE']
        abnormal_data = [element]
        abnormal_data_code = [element_code]
        abnormal_data = np.array(abnormal_data)
        abnormal_data_code = np.array(abnormal_data_code)
        air_abnormal_data.append(abnormal_data)
        air_abnormal_data_code.append(abnormal_data_code)
        air_abnormal_file_name.append(file_name)

    # torch 구조를 맞추기위한 reshape 과정
    air_normal_data = np.array(air_normal_data)
    air_normal_data_code = np.array(air_normal_data_code)
    air_abnormal_data = np.array(air_abnormal_data)
    air_abnormal_data_code = np.array(air_abnormal_data_code)

    # air_normal_data 데이터에 대한 normalize 320시간 간격으로 하기
    for i in range(air_normal_data.shape[0]):
        for j in range(opt.nc):
            air_normal_data[i][j]=normalize(air_normal_data[i][j][:])
    
    # 2년치(대략 15000시간)에 대한 normalize 한번에 하기
    # air_normal_data = normalize(air_normal_data)
    
    # air_abnormal_data normalize 320시간 간격으로
    for i in range(air_abnormal_data.shape[0]):
        for j in range(opt.nc):
            air_abnormal_data[i][j]=normalize(air_abnormal_data[i][j][:])
    air_abnormal_data=air_abnormal_data[:,:opt.nc,:]
    
    # 2년치(대략 15000시간)에 대한 normalize 한번에 하기
    # air_abnormal_data = normalize(air_abnormal_data)

    # train / test
    test_normal_data, test_normal_label, train_normal_data,train_normal_label, test_normal_filename, _ = getFloderK(air_normal_data,opt.folder,0, path=air_normal_file_name)      # normal air data에 대해서는 0으로 라벨링
                                                                                                                                                                            # test_normal_filename: 파일이름이 저장되어 있음
                                                                                                                                                                            # train_normal_filename도 마찬가지
    test_normal_code, test_normal_code_label, train_normal_code, train_normal_code_label = getFloderK(air_normal_data_code, opt.folder, 0)  # 나는 단지 air_normal_code를 양식에 맞게 Dataloader에 넣고 싶은 것임
                                                                                                                                            # 학습에는 얘네 전혀 안쓰임. air_normal_code 값만 살려서 그래프에 띄워주고 싶은것
    test_abnormal_data, test_abnormal_label, test_abnormal_filename = air_abnormal_data, np.ones((air_abnormal_data.shape[0], 1)), air_abnormal_file_name      # abnormal air data에 대해서는 1로 라벨링
    test_abnormal_code , test_abnormal_code_label = air_abnormal_data_code, np.ones((air_abnormal_data_code.shape[0], 1))     # 역시나 얘네도 전혀 쓸모 없음. 단지 air_abnormal_data_code를 살리기 위함

    # train / val
    train_normal_data, val_normal_data, train_normal_label, val_normal_label = getPercent(train_normal_data, train_normal_label, 0.1, 0)
    train_normal_code, val_normal_code, train_normal_code_label, val_normal_code_label = getPercent(train_normal_code, train_normal_code_label, 0.1, 0) # validation에서 일부 testdata를 가져가기 때문에 같은 부분의 code도 가져가게끔 하기 위함. 그래야 맞는 데이터에 맞는 code가 나올 듯
    test_abnormal_data, val_abnormal_data, test_abnormal_label, val_abnormal_label, test_abnormal_filename, val_abnormal_filename = getPercent(test_abnormal_data, test_abnormal_label, 0.1, 0, path=test_abnormal_filename)
    test_abnormal_code, val_abnormal_code, test_abnormal_code_label, val_abnormal_code_label = getPercent(test_abnormal_code, test_abnormal_code_label, 0.1, 0)
    
    assert test_abnormal_data.shape[0] > 250
    np.random.seed(0)
    idx = np.random.randint(0, len(test_abnormal_data), size=100)
    ts_metric_test_abnormal_data = []
    ts_metric_test_abnormal_code = []
    ts_metric_test_abnormal_label = test_abnormal_label[:100]
    for i in idx:
        ts_metric_test_abnormal_data.append(test_abnormal_data[i])
        ts_metric_test_abnormal_code.append(test_abnormal_code[i])
    ts_metric_test_abnormal_data = np.array(ts_metric_test_abnormal_data)
    ts_metric_test_abnormal_code = np.array(ts_metric_test_abnormal_code)
    
    # val_normal_data 에는 air_normal_data와 air_abnormal_data가 같이 들어가 있어야 함
    # validation에서 nan을 뱉어내는 오류를 해결하기 위함
    val_data = np.concatenate([val_normal_data, val_abnormal_data])
    val_label = np.concatenate([val_normal_label, val_abnormal_label])

    print("---------------NORMAL----------------")
    print("train_normal_data size:{}".format(train_normal_data.shape))
    print("train_normal_code size:{}".format(train_normal_code.shape))
    print("test_normal_data size:{}".format(test_normal_data.shape))
    print("test_normal_code size{}".format(test_normal_code.shape))
    print("test_normal_filename len:{}".format(len(test_normal_filename)))
    print("val_normal_data size:{}".format(val_normal_data.shape))
    print("val_normal_code size:{}".format(val_normal_code.shape))
    print("---------------ABNORMAL----------------")
    print("No train data for abnormal")
    print("test_abnormal_data size:{}".format(test_abnormal_data.shape))
    print("test_abnormal_code:{}".format(test_abnormal_code.shape))
    print("test_abnormal_filename len:{}".format(len(test_abnormal_filename)))
    print("val_abnormal_data size:{}".format(val_abnormal_data.shape))
    print("val_abnormal_code size:{}".format(val_abnormal_code.shape))        
    print("---------------NORMAL + ABNORMAL----------------")
    print("val_data size:{}".format(val_data.shape))
    print("---------------For Time Series Metric----------------")
    print("ts_metric_test_abnormal_data:{}".format(ts_metric_test_abnormal_data.shape))
    print("ts_metric_test_abnormal_code:{}".format(ts_metric_test_abnormal_code.shape))
    
    # 2021-01-03
    # 데이터 늘렸을때 nan 발생하는 문제 해결하기 위함
    # 아마 데이터들 중 nan 값이 들어있는듯 함
    
    train_normal_data = np.nan_to_num(train_normal_data)
    val_normal_data = np.nan_to_num(val_normal_data)
    test_normal_data = np.nan_to_num(test_normal_data)
    test_abnormal_data = np.nan_to_num(test_abnormal_data)

    train_dataset = TensorDataset(torch.Tensor(train_normal_data),torch.Tensor(train_normal_label))  # normal만 들어가있음
    test_dataset = TensorDataset(torch.Tensor(test_normal_data),torch.Tensor(test_normal_label)) # normal만 들어가있음
    test_normal_code_dataset = TensorDataset(torch.Tensor(test_normal_code),torch.Tensor(test_normal_code_label))
    test_abnormal_dataset = TensorDataset(torch.Tensor(test_abnormal_data),torch.Tensor(test_abnormal_label))  # abnormal만 들어가있음
    test_abnormal_code_dataset = TensorDataset(torch.Tensor(test_abnormal_code),torch.Tensor(test_abnormal_code_label))
    val_dataset= TensorDataset(torch.Tensor(val_normal_data), torch.Tensor(val_normal_label))    # normal, abnormal 같이 들어가 있음
    val_normal_code_dataset = TensorDataset(torch.Tensor(val_normal_code),torch.Tensor(val_normal_code_label))
    
    # Time-series metric을 위한 TensorDataset
    ts_metric_test_abnormal_data_dataset = TensorDataset(torch.Tensor(ts_metric_test_abnormal_data),torch.Tensor(ts_metric_test_abnormal_label))
    ts_metric_test_abnormal_code_dataset = TensorDataset(torch.Tensor(ts_metric_test_abnormal_code),torch.Tensor(ts_metric_test_abnormal_label))

    dataloader = {"train_normal": DataLoader(      # normal
                    dataset=train_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=True),

                "test_normal": DataLoader(         # normal
                    dataset=test_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "test_normal_code": DataLoader(     
                    dataset=test_normal_code_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),

                "test_abnormal": DataLoader(
                    dataset=test_abnormal_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),

                "test_abnormal_code": DataLoader(
                    dataset=test_abnormal_code_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "val": DataLoader(
                    dataset=val_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "val_normal_code": DataLoader(
                    dataset=val_normal_code_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "ts_metric_test_abnormal_data": DataLoader(
                    dataset=ts_metric_test_abnormal_data_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "ts_metric_test_abnormal_code": DataLoader(
                    dataset=ts_metric_test_abnormal_code_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),
                }
    return dataloader, test_normal_filename, test_abnormal_filename

class MStation():
    def __init__(self, _data, _ele_name, _size):
        self._data = _data
        self._size = _size
        self._ele_name = _ele_name
        self.normData = {}
        self.anormData = {}
        
    def getNorm(self):
        """ Get normal data of normal code range.
        but I don't dropout columns didn't selected.
        For future using

        Returns:
            dict: type of value is DataFrame
        """        
        idx = 0
        for i in range(0, len(self._data), self._size):
            sliced = self._data.iloc[i: i + self._size]
            sliced = sliced[sliced[self._ele_name + '_CODE'] == 0]
        
            if len(sliced) == self._size:
                self.normData[idx] = sliced
                idx += 1
                
        return self.normData
    
    def getAnorm(self):
        idx = 0
        for i in range(0, len(self._data), self._size):
            sliced = self._data.iloc[i: i + self._size]
            
            if len(sliced) == self._size and len(sliced[sliced[self._ele_name + '_CODE'] != 0]) != 0:  # 하나라도 0이 아닌 code가 존재한다면, len(320)은 길이가 320인 데이터가 들어올 수 있도록 강제하기 위함
                self.anormData[idx] = sliced
                idx += 1
                
        return self.anormData
    
def pd_to_class(_ele_name:str, _size:int)->dict:
    """ Change pandas dataframe to Mstation class

    Args:
        _ele_name (str): element name you want to get
        _size (int): length of per data

    Returns:
        dict: It has measuring station id as key,  and has MStation in each id.
    """    
    
    data={}    
    logger = Logger('pd_to_class')
    
    # load pickle
    with gzip.open(os.path.join('data', 'NIER_dataset', 'super_MAIN.pickle'), 'rb') as f:
        ori_data = pickle.load(f)
        
    # delete unimportant columns
    measure_id = ori_data.keys()
    cols = ['MDATETIEM','SO2', 'SO2_CODE', 'CO', 'CO_CODE', 'O3', 'O3_CODE', 'PM10', 'PM10_CODE',
                'PM25', 'PM25_CODE', 'NO', 'NO_CODE', 'NO2', 'NO2_CODE', 'NOX', 'NOX_CODE']
    for idx in measure_id:
        sample = ori_data[idx]
        sample = sample[[col for col in cols]]
        sample.loc['MDATETIEM'] = pd.to_datetime(sample['MDATETIEM'])
        sample.rename(columns = {'MDATETIEM': 'TIME-STAMP'}, inplace = True)
        data[idx] = MStation(sample, _ele_name, _size)      # make MStation class
    
    logger.info("Finished to change pandas to MStation class..")
    return data
    
def load_data_v2(opt, _ele_name:str, _size:int):
    logger = Logger('load_data')
    
    norm_data = {'value':[], 'code':[], 'label':[]}     # label은 만약 데이터가 (3, 320)이면 (3, 1) 형태이면서 정상 구간이였으면 0, 비정상이 포함되어 있으면 1로
    anorm_data = {'value':[],'code':[], 'label':[]}
    
    mstation_set = pd_to_class(_ele_name, _size)
    
    ## get norm_data, anorm_data
    for id in mstation_set.keys():
        for norm in mstation_set[id].getNorm().values():
            norm_data['value'].append(norm[_ele_name].to_numpy())
            norm_data['code'].append(norm[_ele_name+'_CODE'].to_numpy())
            
        for anorm in mstation_set[id].getAnorm().values():            
            anorm_data['value'].append(anorm[_ele_name].to_numpy())
            anorm_data['code'].append(anorm[_ele_name+'_CODE'].to_numpy())
            
    norm_data['value'] = np.array(norm_data['value'])
    norm_data['code'] = np.array(norm_data['code'])
    anorm_data['value'] = np.array(anorm_data['value'])
    anorm_data['code'] = np.array(anorm_data['code'])
    
    logger.info(f"norm_data['value'] {norm_data['value'].shape}, norm_data['code']: {norm_data['code'].shape}")
    logger.info(f"anorm_data['value'] {anorm_data['value'].shape}, anorm_data['code']: {anorm_data['code'].shape}")
    ##

    # norm_data['value'] normalize 320시간 간격으로 하기
    for i in range(norm_data['value'].shape[0]):
        norm_data['value'][i] = normalize(norm_data['value'][i])
            # air_normal_data[i][j]=normalize(air_normal_data[i][j][:])
    
    # 2년치(대략 15000시간)에 대한 normalize 한번에 하기
    # air_normal_data = normalize(air_normal_data)
    
    for i in range(anorm_data['value'].shape[0]):
        anorm_data['value'][i] = normalize(anorm_data['value'][i])
    
    def deleteNan(_value:np.array, _code:np.array)->Union[np.array, np.array]:
        # 이슈. 한 데이터의 구간 값이 모두 같아서, max, min값이 동일해지고 따라서 normalize 할 때 0으로 나누는 현상이 발생함.
        # 이를 제거하기 위해 nan이 들어있는 데이터들은 지우겠음
        deleteIdx = []
        bool_data = np.isnan(_value)
        
        for row in range(len(bool_data)):
            for col in range(len(bool_data[row])):
                if bool_data[row][col] == True:
                    deleteIdx.append(row)
                    break
        _value = np.delete(_value, deleteIdx, 0)
        _code = np.delete(_code, deleteIdx, 0)
        # logger.info(f'Deleted Index: {deleteIdx}')
        # logger.info(f'Deleted data shape: {_value.shape}')
        return _value, _code
    
    def change2Binary(_code):
        """change wrong code to 1

        Args:
            _code (np.array)

        Returns:
            np.array
        """           
        _code = np.where(_code > 0, 1, 0)
        return _code
    
    norm_data['value'], norm_data['code'] = deleteNan(norm_data['value'], norm_data['code'])
    anorm_data['value'], anorm_data['code'] = deleteNan(anorm_data['value'], anorm_data['code'])
    norm_data['code'] = change2Binary(norm_data['code'])
    anorm_data['code'] = change2Binary(anorm_data['code'])
    norm_data['label'] = np.zeros(shape=(norm_data['value'].shape[0], 1))
    anorm_data['label'] = np.ones(shape=(anorm_data['value'].shape[0], 1))
    
    logger.info(f"After delete Nan.   norm_data['value'] {norm_data['value'].shape}, norm_data['code']: {norm_data['code'].shape}, norm_data['label']: {norm_data['label'].shape}")
    logger.info(f"After delete Nan.   anorm_data['value'] {anorm_data['value'].shape}, anorm_data['code']: {anorm_data['code'].shape}, anorm_data['label']: {anorm_data['label'].shape}")
    logger.info("Normalized finished..")
    
    # 2년치(대략 15000시간)에 대한 normalize 한번에 하기
    # air_abnormal_data = normalize(air_abnormal_data)

    # train / val
    train_norm_data, test_norm_data, val_norm_data = {}, {}, {}
    test_anorm_data, val_anorm_data = {}, {}        # anorm_data는 train에서는 안쓰임
    
    train_nv, test_nv, train_nc, test_nc, train_nl, test_nl = train_test_split(norm_data['value'], 
                                                                               norm_data['code'],      # code는 나중에 ts Metric 적용할 때 쓰임. label에는 안쓰여
                                                                               norm_data['label'],
                                                                               test_size=0.2, 
                                                                               shuffle=True) # nv: norm value / nc: norm code
    val_nv, test_nv, val_nc, test_nc, val_nl, test_nl = train_test_split(test_nv, 
                                                        test_nc,
                                                        test_nl, 
                                                        test_size=0.3, 
                                                        shuffle=True)
    
    val_anv, test_anv, val_anc, test_anc, val_anl, test_anl = train_test_split(anorm_data['value'], 
                                                            anorm_data['code'],
                                                            anorm_data['label'],
                                                            test_size=0.08, 
                                                            shuffle=True
                                                            )
    
    ## ts_metric
    assert test_anv.shape[0] > 250
    np.random.seed(0)
    idx = np.random.randint(0, len(test_anv), size=100)
    ts_metric_test_anv = []
    ts_metric_test_anc = []
    ts_metric_test_anl = test_anl[:100]
    for i in idx:
        ts_metric_test_anv.append(test_anv[i])
        ts_metric_test_anc.append(test_anc[i])
    ts_metric_test_anv = np.array(ts_metric_test_anv)
    ts_metric_test_anc = np.array(ts_metric_test_anc)
    
    # val_normal_data 에는 air_normal_data와 air_abnormal_data가 같이 들어가 있어야 함
    # validation에서 nan을 뱉어내는 오류를 해결하기 위함
    print(val_nv.shape)
    print(val_anv.shape)
    val_nv_anv = np.vstack([val_nv, val_anv])      # normal anormal 같이 있으니 이런식으로 변수네이밍
    print(val_nv_anv.shape)
    val_nc_anc = np.vstack([val_nc, val_anc])
    val_nl_anl = np.vstack([val_nl, val_anl])

    logger.info(f"train_nv size: {train_nv.shape}\n" +
                f"val_nv_anv size: {val_nv_anv.shape}\n" + 
                f"test_nv size: {test_nv.shape}\n" + 
                f"test_anv size: {test_anv.shape}")
    
    # 2021-01-03
    # 데이터 늘렸을때 nan 발생하는 문제 해결하기 위함
    # 아마 데이터들 중 nan 값이 들어있는듯 함
    
    # train_normal_data = np.nan_to_num(train_normal_data)
    # val_normal_data = np.nan_to_num(val_normal_data)
    # test_normal_data = np.nan_to_num(test_normal_data)
    # test_abnormal_data = np.nan_to_num(test_abnormal_data)

    # 1차원 데이터는 (Batch, NumofChannel, length) 형식으로 들어가야함
    # 만약 2차원 이미지 데이터였으면 (Batch, Channel, Height, Width)
    train_nv, val_nv, test_nv, val_anv, test_anv = train_nv.reshape(-1, 1, opt.isize), val_nv.reshape(-1, 1, opt.isize), test_nv.reshape(-1, 1, opt.isize), val_anv.reshape(-1, 1, opt.isize), test_anv.reshape(-1, 1, opt.isize)
    val_nv_anv = val_nv_anv.reshape(-1, 1, opt.isize)
    ts_metric_test_anv = ts_metric_test_anv.reshape(-1, 1, opt.isize)
    
    train_dataset = TensorDataset(torch.Tensor(train_nv),torch.Tensor(train_nl))  # normal만 들어가있음. 320길이당 라벨이 하나씩 필요하므로 code를 쓰는게 아닌 np.ones를 씀
    test_dataset = TensorDataset(torch.Tensor(test_nv),torch.Tensor(test_nl)) # normal만 들어가있음
    test_normal_code_dataset = TensorDataset(torch.Tensor(test_nc),torch.Tensor(test_nl))
    test_abnormal_dataset = TensorDataset(torch.Tensor(test_anv),torch.Tensor(test_anl))  # abnormal만 들어가있음
    test_abnormal_code_dataset = TensorDataset(torch.Tensor(test_anc),torch.Tensor(test_anl))
    val_dataset= TensorDataset(torch.Tensor(val_nv_anv), torch.Tensor(val_nl_anl))    # normal, abnormal 같이 들어가 있음
    val_normal_code_dataset = TensorDataset(torch.Tensor(val_nc),torch.Tensor(val_nl))
    
    # Time-series metric을 위한 TensorDataset
    ts_metric_test_abnormal_data_dataset = TensorDataset(torch.Tensor(ts_metric_test_anv),torch.Tensor(ts_metric_test_anl))
    ts_metric_test_abnormal_code_dataset = TensorDataset(torch.Tensor(ts_metric_test_anc),torch.Tensor(ts_metric_test_anl))

    dataloader = {"train_normal": DataLoader(      # normal
                    dataset=train_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=True),        # 코드보니까 여기를 True로 놓아야 bse에서 size 차이 안남. 코드상으로 label을 self.opt.batchsize로 주고있어서, 마지막에 input 데이터 크기가 달라지는거 고려안함

                "test_normal": DataLoader(         # normal
                    dataset=test_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "test_normal_code": DataLoader(     
                    dataset=test_normal_code_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),

                "test_abnormal": DataLoader(
                    dataset=test_abnormal_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),

                "test_abnormal_code": DataLoader(
                    dataset=test_abnormal_code_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "val": DataLoader(
                    dataset=val_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "val_normal_code": DataLoader(
                    dataset=val_normal_code_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "ts_metric_test_abnormal_data": DataLoader(
                    dataset=ts_metric_test_abnormal_data_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "ts_metric_test_abnormal_code": DataLoader(
                    dataset=ts_metric_test_abnormal_code_dataset,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=False,
                    num_workers=int(opt.workers),
                    drop_last=False),
                }
    # return dataloader, test_normal_filename, test_abnormal_filename
    return dataloader

def getFloderK(data, folder, label, path=None):
    """[데이터 나눠주는 함수]

    Args:
        data ([type]): [description]
        folder ([type]): [description]
        label ([type]): [description]
        path ([type], optional): [파일 이름이 옴]. Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        _: 안쓸 filename이기 때문에(내가 filename 쓰는 이유가 png 파일에서 단순 번호가 아니라 파일명에 실어주기 위함인데, train 이미지에 관해서는 png 파일을 안만듬)
    """
    normal_cnt = data.shape[0]
    folder_num = int(normal_cnt / 5)
    folder_idx = folder * folder_num

    folder_data = data[folder_idx:folder_idx + folder_num]
    
    if path != None:
        file_name = path[folder_idx:folder_idx + folder_num]    
        # file_name = np.array(file_name)               # 코딩해보니까 file_name은 numpy가 아니라 list로 존재해야 훨씬 편하더라
        _ = np.concatenate([path[:folder_idx], path[folder_idx + folder_num:]])

    remain_data = np.concatenate([data[:folder_idx], data[folder_idx + folder_num:]])
    
    if label==0:
        folder_data_y = np.zeros((folder_data.shape[0], 1))
        remain_data_y=np.zeros((remain_data.shape[0], 1))
    elif label==1:
        folder_data_y = np.ones((folder_data.shape[0], 1))
        remain_data_y = np.ones((remain_data.shape[0], 1))
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    
    if path != None:
        return folder_data,folder_data_y,remain_data,remain_data_y, file_name, _
    
    return folder_data,folder_data_y,remain_data,remain_data_y

def getPercent(data_x,data_y, test_size,seed, path=None):
    if path != None:
        train_x, test_x, train_y, test_y, train_z, test_z = train_test_split(data_x, data_y, path, test_size=test_size,random_state=seed, shuffle=False)
        return train_x, test_x, train_y, test_y, train_z, test_z
    else:
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=test_size,random_state=seed, shuffle=False)
        return train_x, test_x, train_y, test_y

def get_full_data(dataloader):

    full_data_x=[]
    full_data_y=[]
    for batch_data in dataloader:
        batch_x,batch_y=batch_data[0],batch_data[1]
        batch_x=batch_x.numpy()
        batch_y=batch_y.numpy()

        # print(batch_x.shape)
        # assert False
        for i in range(batch_x.shape[0]):
            full_data_x.append(batch_x[i,0,:])
            full_data_y.append(batch_y[i])

    full_data_x=np.array(full_data_x)
    full_data_y=np.array(full_data_y)
    assert full_data_x.shape[0]==full_data_y.shape[0]
    print("full data size:{}".format(full_data_x.shape))
    return full_data_x,full_data_y


def data_aug(train_x,train_y,times=2):
    res_train_x=[]
    res_train_y=[]
    for idx in range(train_x.shape[0]):
        x=train_x[idx]
        y=train_y[idx]
        res_train_x.append(x)
        res_train_y.append(y)

        for i in range(times):
            x_aug=aug_ts(x)
            res_train_x.append(x_aug)
            res_train_y.append(y)

    res_train_x=np.array(res_train_x)
    res_train_y=np.array(res_train_y)

    return res_train_x,res_train_y

def aug_ts(x):
    left_ticks_index = np.arange(0, 140)
    right_ticks_index = np.arange(140, 319)
    np.random.shuffle(left_ticks_index)
    np.random.shuffle(right_ticks_index)
    left_up_ticks = left_ticks_index[:7]
    right_up_ticks = right_ticks_index[:7]
    left_down_ticks = left_ticks_index[7:14]
    right_down_ticks = right_ticks_index[7:14]

    x_1 = np.zeros_like(x)
    j = 0
    for i in range(x.shape[1]):
        if i in left_down_ticks or i in right_down_ticks:
            continue
        elif i in left_up_ticks or i in right_up_ticks:
            x_1[:, j] =x[:,i]
            j += 1
            x_1[:, j] = (x[:, i] + x[:, i + 1]) / 2
            j += 1
        else:
            x_1[:, j] = x[:, i]
            j += 1
    return x_1
'''
BeatGan preprocess function
'''

import os
import numpy as np
from numpy.lib.function_base import delete
from sklearn.utils import shuffle
import torch
# from torch._C import float32, float64
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

seed = 42
np.random.seed(seed)

def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1

class MStation():
    def __init__(self, _data, _ele_name, _size):
        self._data = _data
        self._size = _size
        self._ele_name = _ele_name
        self.normData = {}
        self.anormData = {}
        
    def getNorm(self):
        """ Get normal data of normal code range.
        but I didn't dropout columns didn't selected.
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
    cols = ['MDATETIEM','AREA_INDEX', 'TIME_INDEX', 'SO2', 'SO2_CODE', 'CO', 'CO_CODE', 'O3', 'O3_CODE', 'PM10', 'PM10_CODE',
                'PM25', 'PM25_CODE', 'NO', 'NO_CODE', 'NO2', 'NO2_CODE', 'NOX', 'NOX_CODE']
    for idx in measure_id:
        sample = ori_data[idx]
        sample = sample[[col for col in cols]]
        sample.loc['MDATETIEM'] = pd.to_datetime(sample['MDATETIEM'])
        sample.rename(columns = {'MDATETIEM': 'TIME-STAMP'}, inplace = True)
        data[idx] = MStation(sample, _ele_name, _size)      # make MStation class
    
    logger.info("Finished to change pandas to MStation class..")
    return data
    
def load_data(opt, _ele_name:str, _size:int, _generated_data):
    logger = Logger('load_data')
    
    norm_data = {'time':[], 'area':[], 'value':[], 'code':[], 'label':[]}     # label은 만약 데이터가 (3, 320)이면 (3, 1) 형태이면서 정상 구간이였으면 0, 비정상이 포함되어 있으면 1로
                                                                              # time, area는 pycaret 평가할때 사용하려고 만듬
    anorm_data = {'time':[], 'area':[], 'value':[], 'code':[], 'label':[]}
    
    mstation_set = pd_to_class(_ele_name, _size)
    
    ## get norm_data, anorm_data
    for id in mstation_set.keys():
        for norm in mstation_set[id].getNorm().values():
            norm_data['time'].append(norm['TIME_INDEX'].to_numpy())
            norm_data['area'].append(norm['AREA_INDEX'].to_numpy())
            norm_data['value'].append(norm[_ele_name].to_numpy(dtype="float64"))        # float64 쓰는 이유: 0으로 나눠줬을 때 프로그램이 중단되지 않고 Nan을 뱉게끔 하기 위함
            norm_data['code'].append(norm[_ele_name+'_CODE'].to_numpy())
        
        for anorm in mstation_set[id].getAnorm().values():
            anorm_data['time'].append(anorm['TIME_INDEX'].to_numpy())
            anorm_data['area'].append(anorm['AREA_INDEX'].to_numpy())            
            anorm_data['value'].append(anorm[_ele_name].to_numpy(dtype="float64"))      # float64 쓰는 이유: 0으로 나눠줬을 때 프로그램이 중단되지 않고 Nan을 뱉게끔 하기 위함
            anorm_data['code'].append(anorm[_ele_name+'_CODE'].to_numpy())
    
    ## 여기서 generated_data와 결합하면 될 듯
    ## generated_data 쓰는 순간, norm_data['time'], norm_data['area']는 고려하지 않기때문에 밑에서 deleteNan 같은거해도 인덱스가 깨짐.
    ## pycaret 평가 할 때는 generated_data 안쓸것기이기 때문에 상관없음
    if _generated_data != None:
        norm_data['time'] = np.vstack([norm_data['time'], _generated_data['time']])     # 'time'에 쓰레기값
        norm_data['area'] = np.vstack([norm_data['area'], _generated_data['area']])     # 'area'에 쓰레기값
        norm_data['value'] = np.vstack([norm_data['value'], _generated_data['value']])
        norm_data['code'] = np.vstack([norm_data['code'], _generated_data['code']])
    
    # change DataFrame to Numpy
    norm_data['time'] = np.array(norm_data['time'])
    norm_data['area'] = np.array(norm_data['area'])
    norm_data['value'] = np.array(norm_data['value'])
    norm_data['code'] = np.array(norm_data['code'])
    # change DataFrame to Numpy
    anorm_data['time'] = np.array(anorm_data['time'])
    anorm_data['area'] = np.array(anorm_data['area'])
    anorm_data['value'] = np.array(anorm_data['value'])
    anorm_data['code'] = np.array(anorm_data['code'])

    norm_data['value'] # normalize 입력데이터 시간 간격으로 하기
    for i in range(norm_data['value'].shape[0]):
        norm_data['value'][i] = normalize(norm_data['value'][i])
    
    # 2년치(대략 15000시간)에 대한 normalize 한번에 하기
    # air_normal_data = normalize(air_normal_data)
    
    for i in range(anorm_data['value'].shape[0]):
        anorm_data['value'][i] = normalize(anorm_data['value'][i])
    
    def deleteNan(_time:np.array, _area:np.array, _value:np.array, _code:np.array)->Union[np.array, np.array, np.array, np.array]:
        # 이슈. 한 데이터의 구간 값이 모두 같아서, max, min값이 동일해지고 따라서 normalize 할 때 0으로 나누는 현상이 발생함.
        # 이를 제거하기 위해 nan이 들어있는 데이터들은 지우겠음
        deleteIdx = []
        bool_data = np.isnan(_value)
        
        for row in range(len(bool_data)):
            for col in range(len(bool_data[row])):
                if bool_data[row][col] == True:
                    deleteIdx.append(row)
                    break
        print(f"지워지는 데이터 갯수: {len(deleteIdx)}")
        _time = np.delete(_time, deleteIdx, 0)
        _area = np.delete(_area, deleteIdx, 0)
        _value = np.delete(_value, deleteIdx, 0)
        _code = np.delete(_code, deleteIdx, 0)
        
        return _time, _area, _value, _code
    
    def change2Binary(_code):
        """change wrong code to 1

        Args:
            _code (np.array)

        Returns:
            np.array
        """           
        _code = np.where(_code > 0, 1, 0)
        return _code
    
    norm_data['time'], norm_data['area'], norm_data['value'], norm_data['code'] = deleteNan(norm_data['time'], norm_data['area'], norm_data['value'], norm_data['code'])
    anorm_data['time'], anorm_data['area'], anorm_data['value'], anorm_data['code'] = deleteNan(anorm_data['time'], anorm_data['area'], anorm_data['value'], anorm_data['code'])
    norm_data['code'] = change2Binary(norm_data['code'])
    anorm_data['code'] = change2Binary(anorm_data['code'])
    norm_data['label'] = np.zeros(shape=(norm_data['value'].shape[0], 1))
    anorm_data['label'] = np.ones(shape=(anorm_data['value'].shape[0], 1))
    
    
    logger.info("Normalize finished and Deleted Nan data..")

    _, test_nt, _, test_na, train_nv, test_nv, train_nc, test_nc, train_nl, test_nl = train_test_split(norm_data['time'],
                                                                                                        norm_data['area'],
                                                                                                        norm_data['value'], 
                                                                                                        norm_data['code'],      # code는 나중에 ts Metric 적용할 때 쓰임. label에는 안쓰여
                                                                                                        norm_data['label'],
                                                                                                        test_size=0.2,
                                                                                                        random_state = seed,  
                                                                                                        shuffle=True) # nv: norm value / nc: norm code
    _, test_nt, _, test_na, val_nv, test_nv, val_nc, test_nc, val_nl, test_nl = train_test_split(test_nt,
                                                                                                 test_na,
                                                                                                 test_nv, 
                                                                                                 test_nc,
                                                                                                 test_nl, 
                                                                                                 test_size=0.3, 
                                                                                                 random_state = seed,  
                                                                                                 shuffle=True)
    
    
    
    _, test_ant, _, test_ana, val_anv, test_anv, val_anc, test_anc, val_anl, test_anl = train_test_split(anorm_data['time'],
                                                                                                         anorm_data['area'],
                                                                                                         anorm_data['value'], 
                                                                                                         anorm_data['code'],
                                                                                                         anorm_data['label'],
                                                                                                         test_size=0.08,     # Normal data와 갯수를 맞춰주기 위함
                                                                                                         random_state = seed,  
                                                                                                         shuffle=True)
    
    def numpySave(norm_time, norm_area, norm_value, norm_code, anorm_time, anorm_area, anorm_value, anorm_code, save=None):
        data = {'norm_time':norm_time, 
                'nore_area':norm_area, 
                'norm_value':norm_value, 
                'norm_code':norm_code, 
                'anorm_time':anorm_time, 
                'anorm_area':anorm_area, 
                'anorm_value':anorm_value, 
                'anorm_code':anorm_code}
        
        if save:
            with gzip.open('./data/NIER_dataset/' + 'pycaret_'+ str(opt.elename) + '_' + str(opt.isize) + '_G' '.pickle', 'wb') as f:
                pickle.dump(data, f)
    
    numpySave(test_nt, test_na, test_nv, test_nc, test_ant, test_ana, test_anv, test_anc, save=True)
    exit()
    
    # val_normal_data 에는 air_normal_data와 air_abnormal_data가 같이 들어가 있어야 함
    # validation에서 nan을 뱉어내는 오류를 해결하기 위함
    
    val_nv_anv = np.vstack([val_nv, val_anv])      # normal anormal 같이 있으니 이런식으로 변수네이밍
    val_nc_anc = np.vstack([val_nc, val_anc])
    val_nl_anl = np.vstack([val_nl, val_anl])

    logger.info(f"train_nv size: {train_nv.shape}\n" +
                f"val_nv_anv size: {val_nv_anv.shape}\n" + 
                f"test_nv size: {test_nv.shape}\n" + 
                f"test_anv size: {test_anv.shape}")

    # 1차원 데이터는 (Batch, NumofChannel, length) 형식으로 들어가야함
    # 만약 2차원 이미지 데이터였으면 (Batch, Channel, Height, Width)
    train_nv, val_nv, test_nv, val_anv, test_anv = train_nv.reshape(-1, 1, opt.isize), val_nv.reshape(-1, 1, opt.isize), test_nv.reshape(-1, 1, opt.isize), val_anv.reshape(-1, 1, opt.isize), test_anv.reshape(-1, 1, opt.isize)
    val_nv_anv = val_nv_anv.reshape(-1, 1, opt.isize)
    # ts_test_anv = ts_test_anv.reshape(-1, 1, opt.isize)
    
    train_nv_set = TensorDataset(torch.Tensor(train_nv),torch.Tensor(train_nl))  # normal만 들어가있음. 320길이당 라벨이 하나씩 필요하므로 code를 쓰는게 아닌 np.ones를 씀
    train_nc_set = TensorDataset(torch.Tensor(train_nc),torch.Tensor(train_nl))
    test_nv_set = TensorDataset(torch.Tensor(test_nv),torch.Tensor(test_nl)) # normal만 들어가있음
    test_nc_set = TensorDataset(torch.Tensor(test_nc),torch.Tensor(test_nl))
    test_anv_set = TensorDataset(torch.Tensor(test_anv),torch.Tensor(test_anl))  # abnormal만 들어가있음
    test_anc_set = TensorDataset(torch.Tensor(test_anc),torch.Tensor(test_anl))
    val_nv_anv_set = TensorDataset(torch.Tensor(val_nv_anv), torch.Tensor(val_nl_anl))    # normal, abnormal 같이 들어가 있음
    val_nc_anc_set = TensorDataset(torch.Tensor(val_nc_anc), torch.Tensor(val_nl_anl))    # normal, abnormal 같이 들어가 있음
    val_nc_set = TensorDataset(torch.Tensor(val_nc),torch.Tensor(val_nl))
    
    # Time-series metric을 위한 TensorDataset
    # ts_test_anv_set = TensorDataset(torch.Tensor(ts_test_anv),torch.Tensor(ts_test_anl))
    # ts_test_anc_set = TensorDataset(torch.Tensor(ts_test_anc),torch.Tensor(ts_test_anl))

    dataloader = {"train_nv_set": DataLoader(      # normal
                    dataset=train_nv_set,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=True),        # 코드보니까 여기를 True로 놓아야 bse에서 size 차이 안남. 코드상으로 label을 self.opt.batchsize로 주고있어서, 마지막에 input 데이터 크기가 달라지는거 고려안함
                  
                  "train_nc_set": DataLoader(      # normal
                    dataset=train_nc_set,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=True),        # 코드보니까 여기를 True로 놓아야 bse에서 size 차이 안남. 코드상으로 label을 self.opt.batchsize로 주고있어서, 마지막에 input 데이터 크기가 달라지는거 고려안함

                "test_nv_set": DataLoader(         # normal
                    dataset=test_nv_set,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "test_nc_set": DataLoader(     
                    dataset=test_nc_set,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),

                "test_anv_set": DataLoader(
                    dataset=test_anv_set,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),

                "test_anc_set": DataLoader(
                    dataset=test_anc_set,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "val_nv_anv_set": DataLoader(
                    dataset=val_nv_anv_set,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "val_nc_anc_set": DataLoader(
                    dataset=val_nc_anc_set,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                "val_nc_set": DataLoader(
                    dataset=val_nc_set,  # torch TensorDataset format
                    batch_size=opt.batchsize,  # mini batch size
                    shuffle=True,
                    num_workers=int(opt.workers),
                    drop_last=False),
                
                # "ts_test_anv_set": DataLoader(
                #     dataset=ts_test_anv_set,  # torch TensorDataset format
                #     batch_size=opt.batchsize,  # mini batch size
                #     shuffle=False,
                #     num_workers=int(opt.workers),
                #     drop_last=False),
                
                # "ts_test_anc_set": DataLoader(
                #     dataset=ts_test_anc_set,  # torch TensorDataset format
                #     batch_size=opt.batchsize,  # mini batch size
                #     shuffle=False,
                #     num_workers=int(opt.workers),
                #     drop_last=False),
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
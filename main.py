from models.beatgan import BeatGAN
import os
import torch
from utils.options import Options
from utils.preprocess import load_data, load_data
from utils.log import Logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    logger = Logger('main')
    logger.info('Main.py started..')    
    
    # from dcgan import DCGAN as myModel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt = Options().parse()
    
    # dataloader, test_normal_filename, test_abnormal_filename = load_data(opt)
    dataloader = load_data(opt, _ele_name = 'PM10', _size = opt.isize)
    
    logger.info("Load data success..")

    if opt.model == "beatgan":
        from models.beatgan import BeatGAN
        model = BeatGAN(opt, dataloader, device)

    else:
        raise Exception("no this model :{}".format(opt.model))

    if not opt.istest:
        logger.info("Train start...")
        model.train()
    else:
        logger.info("Test start...")
        model.load()        # BeatGan이 상속하고 있는 (상속하다 == 재산을 물려받다) AD_MODEL에 있는 load 함수    
        
        if opt.ts:
            model.ts_test()
        else:
            model.ori_test()
        
        # model.test_time()
        # model.plotTestFig()
        # print("threshold:{}\tf1-score:{}\tauc:{}".format( th, f1, auc))
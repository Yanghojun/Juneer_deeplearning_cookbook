import sys
import os
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from options import Options
from preprocess import load_data
from utils.log import Logger

if __name__ == '__main__':
    logger = Logger('data_generator')
    logger.info('Data_generator.py started..')
    
    # from dcgan import DCGAN as myModel
    device = torch.device("cuda:0" if
    torch.cuda.is_available() else "cpu")

    opt = Options().parse()
    
    generated_data = None
    
    dataloader = load_data(opt, _ele_name=opt.elename, _size=opt.isize, _generated_data=generated_data)        # 필요없는 부분이긴 한데 BeatGAN 클래스 인자로 dataloader 줘야하니 그냥 두자
    print("load data success!!!")

    if opt.model == "beatgan":
        from models.beatgan import BeatGAN

    else:
        raise Exception("no this model :{}".format(opt.model))


    model=BeatGAN(opt,dataloader,device)
    
    print("################  Generating  ##################")
    model.load()        # BeatGan이 상속하고 있는 (상속하다 == 재산을 물려받다) AD_MODEL에 있는 load 함수
    fake_data = model.generate()
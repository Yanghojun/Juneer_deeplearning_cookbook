import argparse
import os
import torch

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset', default='NIER', help='normal json files of sensor')
        # self.parser.add_argument('--dataroot', default='data/NIER_dataset/PM10/', help='path to dataset, ex) dataset/PM10/')
        self.parser.add_argument('--elename', type=str, default='PM10', help='choose what you want to use for data (It is used to train and could be also used to test)')
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
        self.parser.add_argument('--isize', type=int, default=320, help='input sequence size.')
        self.parser.add_argument('--nc', type=int, default=1, help='input sequence channels')
        self.parser.add_argument('--nz', type=int, default=50, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=32)
        self.parser.add_argument('--ndf', type=int, default=32)
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--model', type=str, default='beatgan', help='choose model')
        self.parser.add_argument('--outf', default='./result', help='output folder')
        self.parser.add_argument('--generated', action='store_true',help='Use generated data or not')

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='parameter')
        self.parser.add_argument('--folder', type=int, default=0, help='folder index 0-4')
        self.parser.add_argument('--n_aug', type=int, default=0, help='aug data times')

        ## Test
        self.parser.add_argument('--istest', action='store_true',help='train model or test model')
        self.parser.add_argument('--ts', action='store_true',help='train model or test model')
        self.parser.add_argument('--threshold', type=float, default=0.05, help='threshold score for anomaly')
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args() 

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])
            
        args = vars(self.opt)       # vars()??? argparse ????????? ?????? ???????????? ????????? ?????? ????????? key, value ????????? ?????????

        # save to the disk
        self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'used_options.txt')
        with open(file_name, 'wt') as f:
            f.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                f.write('%s: %s\n' % (str(k), str(v)))
            f.write('-------------- End ----------------\n')
        return self.opt
    
if __name__ == '__main__':
    opt = Options().parse()
    
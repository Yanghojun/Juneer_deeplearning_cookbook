from numpy.lib.npyio import save
from utils import metric as ts_metric       # 이거 from ..utils import metric as ts_metric으로 구성해야 할 줄 알았는데 main에서 실행하는것이니 경로를 main.py 기준으로 짜야하는듯?
                                            # print(sys.path)를 통해서 main.py가 중심임을 알 수 있음.
                                            
# from metric import evaluate       beatgan 원래 평가코드인데 지금은 tsmetric 쓰니까 필요없을듯
from utils.metric import beatgan_ori_evaluate
import time
import os
import sys
import pickle
import numpy as np
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.plot import *

dirname = os.path.dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))


class Encoder(nn.Module):
    def __init__(self, ngpu, opt, out_z):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Sequential(
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv6 = nn.Conv1d(opt.ndf * 16, out_z, 10, 1, 0, bias=False)

        self.main = nn.Sequential(
            # input is (nc) x 320
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 160
            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 80
            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 40
            nn.Conv1d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 20
            nn.Conv1d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 10

            nn.Conv1d(opt.ndf * 16, out_z, 10, 1, 0, bias=False),
            # state size. (nz) x 1
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            # output = self.main(input)
            output = self.conv1(input)
            output = self.conv2(output)
            output = self.conv3(output)
            output = self.conv4(output)
            output = self.conv5(output)
            output = self.conv6(output)
        return output

##


class Decoder(nn.Module):
    def __init__(self, ngpu, opt):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(opt.nz, opt.ngf*16, 10, 1, 0, bias=False),
            nn.BatchNorm1d(opt.ngf*16),
            nn.ReLU(True),
            # state size. (ngf*16) x10
            nn.ConvTranspose1d(opt.ngf * 16, opt.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 20
            nn.ConvTranspose1d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 40
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf*2),
            nn.ReLU(True),
            # state size. (ngf) x 80
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 160
            nn.ConvTranspose1d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 320


        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class AD_MODEL(object):
    def __init__(self, opt, dataloader, device):
        self.G = None
        self.D = None

        self.opt = opt
        self.niter = opt.niter
        self.dataset = opt.dataset
        self.model = opt.model
        self.outf = opt.outf

    def train(self):
        raise NotImplementedError

    def visualize_results(self, epoch, samples, is_train=True):
        if is_train:
            sub_folder = "train"
        else:
            sub_folder = "test"

        save_dir = os.path.join(self.outf, self.model,
                                self.dataset, sub_folder)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_plot_sample(samples, epoch, self.dataset, num_epochs=self.niter,
                         impath=os.path.join(save_dir, 'epoch%03d' % epoch + '.png'))

    def visualize_pair_results(self, epoch, samples1, samples2, is_train=True):
        if is_train:
            sub_folder = "train"
        else:
            sub_folder = "test"

        save_dir = os.path.join(self.outf, self.model,
                                self.dataset, sub_folder)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_plot_pair_sample(samples1, samples2, epoch, self.dataset, num_epochs=self.niter,
                              impath=os.path.join(save_dir, 'epoch%03d' % epoch + '.png'))

    def save(self, train_hist):
        save_dir = os.path.join(self.outf, self.model, self.dataset, "model")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, self.model + '_history.pkl'), 'wb') as f:
            pickle.dump(train_hist, f)

    def save_weight_GD(self):
        save_dir = os.path.join(self.outf, self.model, self.dataset, "model")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(
            save_dir, self.model+"_folder_"+str(self.opt.folder) + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(
            save_dir, self.model+"_folder_"+str(self.opt.folder) + '_D.pkl'))

    def load(self):
        save_dir = os.path.join(self.outf, self.model, self.dataset, "model")

        self.G.load_state_dict(torch.load(os.path.join(
            save_dir, self.model+"_folder_"+str(self.opt.folder) + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(
            save_dir, self.model+"_folder_"+str(self.opt.folder) + '_D.pkl')))

    def save_loss(self, train_hist):
        loss_plot(train_hist, os.path.join(
            self.outf, self.model, self.dataset), self.model)

    def saveTestPair(self, pair, code, save_dir, filename=None):
        '''

        :param pair: list of (input,output)
        :param save_dir:

        :return:
        '''

        assert save_dir is not None
        if filename == None:
            for idx, p in enumerate(zip(pair, code)):
                input = p[0][0]
                output = p[0][1]
                _code = p[1][0]

                save_pair_fig(input, output, _code, os.path.join(
                    save_dir, str(idx)))
        else:
            for idx, p in enumerate(zip(pair, code, filename)):
                input = p[0][0]
                output = p[0][1]
                _code = p[1][0]
                # filename = p[2].split("\\")[1][:-5]
                filename = p[2]

                save_pair_fig(input, output, _code, os.path.join(
                    save_dir, filename + ".png"))

    def analysisRes(self, N_res, A_res, min_score, max_score, threshold, save_dir):
        '''
        dist0_NA.png, dist0_Nair_abnormal.png, logdist0_NA.png, logdist0_Nair_abnormal.png 파일 만드는 함수        
        TP, FP, TN, FN 이거 출력도 여기서 해줌
        res_th는 opt.threshold 값. default = 0.05
        normal 데이터의 error, abnormal 데이터의 error를 같이 보냄
        
        :param N_res: list of normal score
        :param A_res:  dict{ "S": list of S score, "V":...}
        :param min_score:
        :param max_score:
        :return:
        '''
        print("############   Analysis   #############")
        print("############   Threshold:{}   #############".format(threshold))
        all_abnormal_score = []
        all_normal_score = np.array([])
        for a_type in A_res:
            a_score = A_res[a_type]
            print("*********  Type:{}  *************".format(a_type))
            normal_score = normal(N_res, min_score, max_score)
            abnormal_score = normal(a_score, min_score, max_score)
            all_abnormal_score = np.concatenate(
                (all_abnormal_score, np.array(abnormal_score)))
            all_normal_score = normal_score
            plot_dist(normal_score, abnormal_score, "Score of normal", a_type,
                      save_dir)

            TP = np.count_nonzero(abnormal_score >= threshold)
            FP = np.count_nonzero(normal_score >= threshold)
            TN = np.count_nonzero(normal_score < threshold)
            FN = np.count_nonzero(abnormal_score < threshold)
            print("TP:{}".format(TP))
            print("FP:{}".format(FP))
            print("TN:{}".format(TN))
            print("FN:{}".format(FN))
            print("Accuracy:{}".format((TP + TN) * 1.0 / (TP + TN + FP + FN)))
            print("Precision:{}".format(TP * 1.0 / (TP + FP)))
            print("Recall:{}".format(TP * 1.0 / (TP + FN)))
            print("specificity:{}".format(TN * 1.0 / (TN + FP)))
            print("F1:{}".format(2.0 * TP / (2 * TP + FP + FN)))

        # all_abnormal_score=np.reshape(np.array(all_abnormal_score),(-1))
        # print(all_abnormal_score.shape)
        plot_dist(all_normal_score, all_abnormal_score, str(self.opt.folder)+"_"+"N", "A",
                  save_dir)


class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        model = Encoder(opt.ngpu, opt, 1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])         # Asterisk(*) 붙인 이유
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


##
class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(opt.ngpu, opt, opt.nz)
        self.decoder = Decoder(opt.ngpu, opt)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_x = self.decoder(latent_i)
        return gen_x, latent_i


class BeatGAN(AD_MODEL):

    def __init__(self, opt, dataloader, device):
        super(BeatGAN, self).__init__(opt, dataloader, device)
        self.dataloader = dataloader
        self.device = device
        self.opt = opt

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator(opt).to(device)
        self.G.apply(weights_init)
        if not self.opt.istest:
            print_network(self.G)

        self.D = Discriminator(opt).to(device)
        self.D.apply(weights_init)
        if not self.opt.istest:
            print_network(self.D)

        # self.bce_criterion = nn.BCELoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.mse_criterion = nn.MSELoss()

        self.optimizerD = optim.Adam(
            self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(
            self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.total_steps = 0
        self.cur_epoch = 0

        self.input = torch.empty(size=(
            self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(
            size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,),
                              dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(
            self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label = 0

        self.out_d_real = None
        self.feat_real = None

        self.fake = None
        self.latent_i = None
        self.out_d_fake = None
        self.feat_fake = None

        self.err_d_real = None
        self.err_d_fake = None
        self.err_d = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print("Train model.")
        start_time = time.time()
        best_auc = 0
        best_auc_epoch = 0

        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):
                self.cur_epoch += 1
                self.train_epoch()
                auc, th, f1 = self.validate()
                ### if auc > best_auc:        # 2021-05-27 저장이 안되길래 if문 빼버림
                best_auc = auc
                best_auc_epoch = self.cur_epoch
                # self.save_weight_GD()
                ##

                f.write("[{}] auc:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(
                    self.cur_epoch, auc, best_auc, best_auc_epoch))
                print("[{}] auc:{:.4f} th:{:.4f} f1:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(
                    self.cur_epoch, auc, th, f1, best_auc, best_auc_epoch))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.niter,
                                                                        self.train_hist['total_time'][0]))

        # 2021-05-27
        # 용량이 커서 에러가 나는건가 해서 에폭당 저장하는게 아니라 맨 마지막에폭만 저장하게끔 수정
        self.save_weight_GD()

        self.save(self.train_hist)

        self.save_loss(self.train_hist)

    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        self.D.train()
        epoch_iter = 0
        for data in self.dataloader["train_nv_set"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1

            self.set_input(data)
            self.optimize()

            errors = self.get_errors()

            self.train_hist['D_loss'].append(errors["err_d"])
            self.train_hist['G_loss'].append(errors["err_g"])

            if (epoch_iter % self.opt.print_freq) == 0:

                print("Epoch: [%d] [%4d/%4d] D_loss(R/F): %.6f/%.6f, G_loss: %.6f" %
                      ((self.cur_epoch), (epoch_iter), self.dataloader["train_nv_set"].dataset.__len__() // self.batchsize,
                       errors["err_d_real"], errors["err_d_fake"], errors["err_g"]))
                # print("err_adv:{}  ,err_rec:{}  ,err_enc:{}".format(errors["err_g_adv"],errors["err_g_rec"],errors["err_g_enc"]))

        self.train_hist['per_epoch_time'].append(
            time.time() - epoch_start_time)
        with torch.no_grad():
            real_input, fake_output = self.get_generated_x()

            # 2020-12-03 일단 돌리는게 급선무니 이건 나중에
            # self.visualize_pair_results(self.cur_epoch,
            #                             real_input,
            #                             fake_output,
            #                             is_train=True)

    def set_input(self, input):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])

        # fixed input for view
        if self.total_steps == self.opt.batchsize:
            with torch.no_grad():
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##

    def optimize(self):

        self.update_netd()
        self.update_netg()

        # If D loss too low, then re-initialize netD
        if self.err_d.item() < 5e-6:
            self.reinitialize_netd()

    def update_netd(self):
        ##

        self.D.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        # self.out_d_real: real data에 대한 D의 output,  self.feat_real: 인코더 결과
        self.out_d_real, self.feat_real = self.D(self.input)
        # encoder, decoder를 사용하는 이유 -> input node보다 hidden node수가 적은데 decoder 쪽에서는 줄어든 hidden node
        # 만을 보고 처음의 input node를 재현해야함. 그럴려면 제일 중요한 feature들을 학습하여야 함. (일단 비지도학습이기 때문에 label도 없음)
        # 제일 중요한 feature를 학습시키기 위함임
        # --
        # Train with fake
        self.label.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        # self.fake: Generator를 통해 나온 가짜데이터(1,8,1440) self.latent_i: ??
        self.fake, self.latent_i = self.G(self.input)
        # self.out_d_fake: 가짜데이터에 대한 것이므로 0,  self.feat_fake: 인코더 결과
        self.out_d_fake, self.feat_fake = self.D(self.fake)
        # --

        self.err_d_real = self.bce_criterion(self.out_d_real, torch.full(
            (self.batchsize,), self.real_label, device=self.device, dtype=torch.float32))
        # self.err_d_real = self.bce_criterion(self.out_d_real, torch.Tensor((self.batchsize,), self.real_label, device=self.device))
        self.err_d_fake = self.bce_criterion(self.out_d_fake, torch.full(
            (self.batchsize,), self.fake_label, device=self.device, dtype=torch.float32))

        self.err_d = self.err_d_real+self.err_d_fake
        self.err_d.backward()
        self.optimizerD.step()

    ##
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.D.apply(weights_init)
        print('Reloading d net')

    ##
    def update_netg(self):
        self.G.zero_grad()
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.fake, self.latent_i = self.G(self.input)
        # self.out_g: Discriminator가 fake data로 output 낸 거. self.feat_fake: fake data에 대한 압축된 data
        self.out_g, self.feat_fake = self.D(self.fake)
        _, self.feat_real = self.D(self.input)

        # self.err_g_adv = self.bce_criterion(self.out_g, self.label)   # loss for ce
        self.err_g_adv = self.mse_criterion(
            self.feat_fake, self.feat_real)  # loss for feature matching
        self.err_g_rec = self.mse_criterion(
            self.fake, self.input)  # constrain x' to look like x

        self.err_g = self.err_g_rec + self.err_g_adv * self.opt.w_adv
        self.err_g.backward()
        self.optimizerG.step()

    ##

    def get_errors(self):

        errors = {'err_d': self.err_d.item(),
                  'err_g': self.err_g.item(),
                  'err_d_real': self.err_d_real.item(),
                  'err_d_fake': self.err_d_fake.item(),
                  'err_g_adv': self.err_g_adv.item(),
                  'err_g_rec': self.err_g_rec.item(),
                  }

        return errors

        ##

    def get_generated_x(self):
        fake = self.G(self.fixed_input)[0]

        return self.fixed_input.cpu().data.numpy(), fake.cpu().data.numpy()

    ##

    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        # time-series
        # y_, y_pred, _, _ = self.predict(
        #     self.dataloader["val"], self.dataloader["val_normal_code"], scale=False)
        
        y_, y_pred = self.predict(
            self.dataloader["val_nv_anv_set"], self.dataloader["val_nc_set"], scale=False)
        
        rocprc, rocauc, best_th, best_f1 = beatgan_ori_evaluate(y_, y_pred)
        return rocauc, best_th, best_f1

    def predict(self, dataloader_, dataloader_code_, scale=True):
        """ 모델 예측값 반환하는 함수

        Args:
            dataloader_ ([type]): [description]
            dataloader_code_ ([type]): 시계열 데이터 평가를 위한 라벨 구축에 필요
            scale (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        with torch.no_grad():

            self.an_scores = torch.zeros(
                size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(
                size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i = torch.zeros(size=(
                len(dataloader_.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.timeseries_metric_score = torch.zeros(size=(len(
                dataloader_.dataset), self.opt.isize), dtype=torch.float32, device=self.device)
            self.timeseries_metric_label = torch.zeros(size=(len(
                dataloader_.dataset), self.opt.isize), dtype=torch.float32, device=self.device)
            self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
                                        device=self.device)

            # dataloader_: normal로 이루어진 데이터셋
            for i, _data in enumerate(zip(dataloader_, dataloader_code_), 0):
                # _data[0]에는 실제 센서값과, 해당 구간이 정상(0)인지 비정상(1)인지 알려주는 레이블도 같이 들어있음
                data = _data[0]
                wrong_code = _data[1][0]
                # self.input, self.gt에 각각 실제 데이터 들어가게 해줌
                self.set_input(data)
                # self.fake: 가짜 데이터, latent_i: Encoder 결과 (압축된 Tensor)
                self.fake, latent_i = self.G(self.input)

                # error = torch.mean(torch.pow((d_feat.view(self.input.shape[0],-1)-d_gen_feat.view(self.input.shape[0],-1)), 2), dim=1)
                #
                error = torch.mean(
                    torch.pow((self.input.view(
                        self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                # 새로운 metric을 위해 활용하는 변수
                error_for_timeseries_metric = torch.pow((self.input.view(
                    self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2)
                error_for_timeseries_label = wrong_code.view(
                    self.input.shape[0], -1)

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize +
                               error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize +
                               error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i[i*self.opt.batchsize: i*self.opt.batchsize +
                              error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)

                # 새로운 metric을 위해 활용하는 변수
                self.timeseries_metric_score[i*self.opt.batchsize: i*self.opt.batchsize +
                                             error_for_timeseries_metric.size(0), :] = error_for_timeseries_metric
                self.timeseries_metric_label[i*self.opt.batchsize: i*self.opt.batchsize +
                                             error_for_timeseries_label.size(0), :] = error_for_timeseries_label

            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                    (torch.max(self.an_scores) - torch.min(self.an_scores))

            y_ = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()

            # 새로운 metric을 위해 활용하는 변수
            y_pred_for_timeseries_metric = self.timeseries_metric_score.cpu().numpy()
            y_pred_for_timeseries_label = self.timeseries_metric_label.cpu().numpy()

            return y_, y_pred,y_pred_for_timeseries_metric, y_pred_for_timeseries_label

    def draw_test_result(self, dataloader_, dataloader_code_, min_score, max_score, threshold, save_dir, filename=None):
        """테스트 결과 그림 그려주는 함수. False(0)가 정상. Positive(1)가 비정상

        Args:
            dataloader_ ([type]): [description]
            dataloader_code_ ([type]): [description]
            min_score ([type]): [description]
            max_score ([type]): [description]
            threshold ([type]): [description]
            save_dir ([type]): [description]
            filename ([type], optional): [description]. Defaults to None.
        """        
        assert save_dir is not None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            # Create big error tensor for the test set.
            test_pair = []
            code = []  # saveTestPair에 같이 보낼 code 담을 리스트
            chosen_filename = []  # saveTestPair에 같이 보낼 filename 담을 리스트

            self.an_scores = torch.zeros(
                size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)

            # for i, data in enumerate(dataloader_, 0):
            for i, data in enumerate(zip(dataloader_, dataloader_code_), 0):
                self.set_input(data[0])     # data[0]은 센서값이랑 라벨링
                # data[1]은 wrongcode랑 라벨링. 근데 wrongcode의 라벨링은 쓰잘데기 없음

                self.fake, latent_i = self.G(self.input)

                error = torch.mean(
                    torch.pow((self.input.view(
                        self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize +
                               error.size(0)] = error.reshape(error.size(0))

                # # Save test images.

                batch_input = self.input.cpu().numpy()
                batch_output = self.fake.cpu().numpy()
                batch_code = data[1][0].cpu().numpy()
                
                if filename == None:
                    pass
                else:
                    batch_filename = filename[data[0][0].shape[0]
                                          * i:data[0][0].shape[0]*i+data[0][0].shape[0]]

                ano_score = error.cpu().numpy()
                
                assert batch_output.shape[0] == batch_input.shape[0] == ano_score.shape[0]
                
                tp_dir = os.path.split(save_dir)[-1]    # 파일 이름에 따라서 어떤 그림을 만들지 정하기 위함.
                for idx in range(batch_input.shape[0]):
                    # if len(test_pair) > 100:  # 100장의 이상의 png 파일을 만들지 않기 위함
                    #     break
                    normal_score = (ano_score[idx]-min_score)/(max_score-min_score)

                    if normal_score<=threshold and tp_dir == "Normal_Data_Predicted_As_Normal - (TN)":
                        test_pair.append((batch_input[idx], batch_output[idx]))
                        code.append(batch_code[idx])
                        
                    elif normal_score>threshold and tp_dir == "Normal_Data_Predicted_As_Anormal - (FP)":
                        test_pair.append((batch_input[idx], batch_output[idx]))
                        code.append(batch_code[idx])
                        
                    elif normal_score<=threshold and tp_dir == "Anormal_Data_Predicted_As_Normal - (FN)":
                        test_pair.append((batch_input[idx], batch_output[idx]))
                        code.append(batch_code[idx])
                        
                    elif normal_score>threshold and tp_dir == "Anormal_Data_Predicted_As_Anormal - (TP)":
                        test_pair.append((batch_input[idx], batch_output[idx]))
                        code.append(batch_code[idx])
                    
                    if filename == None:
                        pass
                    else:
                        chosen_filename.append(batch_filename[idx])

            # print(len(test_pair))
            if filename == None:
                self.saveTestPair(pair=test_pair, code=code, save_dir=save_dir)
            else:    
                self.saveTestPair(pair=test_pair, code=code, save_dir=save_dir, filename=chosen_filename)
                
                

    def ts_evaluation(self, diff_of_score, label, threshold):
        """[타임시리즈 평가법으로 성능 평가하는 함수]

        Args:
            diff_of_score ([type]): [BeatGan 모델을 통해 진짜 데이터와 가짜 데이터의 차이값을 리스트로 가지고 있음]
            label ([type]): [진짜 데이터의 레이블]
            threshold ([type]): [어느 정도의 차이를 비정상으로 여길지 결정하는 수]
        """

        # 데이터 전처리
        binary = np.where(diff_of_score > threshold, 1, 0)
        # 다양한 코드들이 존재하는데 0, 1로 이진화 하기 위함
        gt_label = np.where(label != 0, 1, 0)

        num_label = gt_label.shape[0]
        precision_middle, recall_middle, f1_middle = 0, 0, 0
        for idx in range(len(gt_label)):
            if(gt_label[idx].sum() == len(gt_label[idx]) or
               binary[idx].sum() == len(binary[idx]) or
               binary[idx].sum() == 0):     # 모델이 모두 0 혹은 모두 1로 예측하거나, ground truth가 모두 1일 경우에는 ts metric assertion에 걸림. 0, 1 이 모두 있어야 해서..
                # 그래서 언급한 경우는 계산에서 제외시킴
                num_label -= 1
                continue

            tp_precision_middle, tp_recall_middle, tp_f1_middle = ts_metric.evaluate(
                gt_label[idx], binary[idx])
            precision_middle += tp_precision_middle
            recall_middle += tp_recall_middle
            f1_middle += tp_f1_middle

        print(f'계산된 num_label: {num_label}')
        precision_middle = precision_middle/num_label
        recall_middle = recall_middle/num_label
        f1_middle = f1_middle/num_label

        print("#" * 15 + "최종" + "#" * 15)
        print("precision: ", precision_middle, "recall: ",
              recall_middle, "f1: ", f1_middle)
        # precision_middle, recall_middle, f1_middle = ts_metric.evaluate(gt_label, binary)    # 리스트 2개로 넘겨줘야함
    def ori_test(self):
        self.G.eval()
        self.D.eval()
        res_th = self.opt.threshold
        save_dir = os.path.join(self.outf, self.model,
                                self.dataset, "test", str(self.opt.folder))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        y_air, y_pred_air, _, _ = self.predict(self.dataloader["test_nv_set"],
                                         self.dataloader["test_nc_set"],
                                         scale=False)
        y_air_abnormal, y_pred_air_abnormal, _, _ = self.predict(self.dataloader["test_anv_set"],
                                                           self.dataloader["test_anc_set"],
                                                           scale=False)
        
        over_all = np.concatenate([y_pred_air, y_pred_air_abnormal])
        over_all_gt = np.concatenate([y_air, y_air_abnormal])
        min_score, max_score = np.min(over_all), np.max(over_all)

        A_res = {
            "Score_of_abnormal": y_pred_air_abnormal,
        }

        
        self.analysisRes(y_pred_air, A_res, min_score,max_score, res_th, save_dir) 
        
        self.draw_test_result(self.dataloader["test_nv_set"], 
                               self.dataloader["test_nc_set"],
                               min_score, 
                               max_score, 
                               res_th, 
                               save_dir=os.path.join(save_dir, "Normal_Data_Predicted_As_Normal - (TN)"))
        
        self.draw_test_result(self.dataloader["test_nv_set"], 
                               self.dataloader["test_nc_set"],
                               min_score, 
                               max_score, 
                               res_th, 
                               save_dir=os.path.join(save_dir, "Normal_Data_Predicted_As_Anormal - (FP)"))  
        
        self.draw_test_result(self.dataloader["test_anv_set"], 
                               self.dataloader["test_anc_set"], 
                               min_score,
                               max_score, 
                               res_th, 
                               save_dir=os.path.join(save_dir, "Anormal_Data_Predicted_As_Normal - (FN)"))
        
        self.draw_test_result(self.dataloader["test_anv_set"], 
                               self.dataloader["test_anc_set"], 
                               min_score,
                               max_score, 
                               res_th, 
                               save_dir=os.path.join(save_dir, "Anormal_Data_Predicted_As_Anormal - (TP)"))
        
        

        aucprc, aucroc, best_th, best_f1 = beatgan_ori_evaluate(
            over_all_gt, (over_all-min_score)/(max_score-min_score))
        print("#############################")
        print("########  Result  ###########")
        print("ap:{}".format(aucprc))
        print("auc:{}".format(aucroc))
        print("best th:{} --> best f1:{}".format(best_th, best_f1))

        with open(os.path.join(save_dir, "res-record.txt"), 'w') as f:
            f.write("auc_prc:{}\n".format(aucprc))
            f.write("auc_roc:{}\n".format(aucroc))
            f.write("best th:{} --> best f1:{}".format(best_th, best_f1))

    def test_time(self):
        self.G.eval()
        self.D.eval()
        size = self.dataloader["test_N"].dataset.__len__()
        start = time.time()

        for i, (data_x, data_y) in enumerate(self.dataloader["test_N"], 0):
            input_x = data_x
            for j in range(input_x.shape[0]):
                input_x_ = input_x[j].view(
                    1, input_x.shape[1], input_x.shape[2]).to(self.device)
                gen_x, _ = self.G(input_x_)

                error = torch.mean(
                    torch.pow(
                        (input_x_.view(input_x_.shape[0], -1) - gen_x.view(gen_x.shape[0], -1)), 2),
                    dim=1)

        end = time.time()
        print((end-start)/size)

    def ts_test(self):
        self.G.eval()
        self.D.eval()
        res_th = self.opt.threshold
        
        # _, _, score, label = self.predict(self.dataloader["ts_test_anv_set"],
        #                                   self.dataloader["ts_test_anc_set"],
        #                                   scale=False)
        
        _, _, score, label = self.predict(self.dataloader["test_anv_set"],
                                          self.dataloader["test_anc_set"],
                                          scale=False)

        # Timeseries metric 함수
        print("Threshold: {}".format(res_th))
        self.ts_evaluation(score, label, res_th)

        # precision_middle, recall_middle, f1_middle = evaluate(val_real, val_pred)
        # with open('report_test_report.txt', 'w') as f:
        #     f.write('\n===============================\n')
        #     f.write(f'precision: {precision_middle}\n')
        #     f.write(f'recall: {recall_middle}\n')
        #     f.write(f'f1: {f1_middle}\n')

    def generate(self, filename):

        self.G.eval()
        self.D.eval()
        res_th = self.opt.threshold
        save_dir = os.path.join(self.outf, self.model,
                                self.dataset, "test", str(self.opt.folder))

        dataloader_ = self.dataloader["test_nv_set"]
        dataloader_code_ = self.dataloader["test_nc_set"]

        with torch.no_grad():
            self.an_scores = torch.zeros(
                size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(
                size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i = torch.zeros(size=(
                len(dataloader_.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.timeseries_metric_score = torch.zeros(size=(len(
                dataloader_.dataset), self.opt.isize), dtype=torch.float32, device=self.device)
            self.timeseries_metric_label = torch.zeros(size=(len(
                dataloader_.dataset), self.opt.isize), dtype=torch.float32, device=self.device)
            self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
                                        device=self.device)

            if not os.path.isdir("./GeneratedData"):
                os.mkdir("./GeneratedData")

            fake_data = []
            # 데이터 이름(PM10, CO, O3 등등)
            data_name = ((self.opt.dataroot).split('/'))[-2]

            if not os.path.isdir(os.path.join("./GeneratedData", data_name)):
                os.mkdir(os.path.join("./GeneratedData", data_name))

            # dataloader_: normal로 이루어진 데이터셋
            for i, _data in enumerate(zip(dataloader_, dataloader_code_), 0):
                # _data[0]에는 실제 센서값과, 해당 구간이 정상(0)인지 비정상(1)인지 알려주는 레이블도 같이 들어있음
                data = _data[0]
                wrong_code = _data[1][0]
                # self.input, self.gt에 각각 실제 데이터 들어가게 해줌
                self.set_input(data)
                # self.fake: 가짜 데이터, latent_i: Encoder 결과 (압축된 Tensor)
                self.fake, latent_i = self.G(self.input)

                fake_data.append(self.fake)

                # 여기부터 복붙코드
                batch_output = self.fake.cpu().numpy()
                batch_code = data[1][0].cpu().numpy()

                # batch_filename = filename[data[0][0].shape[0]*i:data[0][0].shape[0]*i+data[0][0].shape[0]] 이거 코드 좀 이상한데..? 배치가 8이면 8개씩 가져와야 할텐데..
                batch_filename = filename[i * batch_output.shape[0]
                    :i * batch_output.shape[0] + batch_output.shape[0]]
                for idx in range(batch_output.shape[0]):
                    '''
                    input data의 파일이름을 살려서 Json 데이터로 만들기
                    '''
                    json_data = {}
                    json_data["area"] = int(batch_filename[idx].split("_")[0])
                    json_data["start_date"] = batch_filename[idx].split("_")[1]
                    json_data["data"] = {
                        f"{data_name}": [float(value) for value in batch_output[idx][0]],
                        f"{data_name}_CODE": [0 for i in range(self.opt.isize)]
                    }
                    with open(os.path.join("./GeneratedData", data_name, "f_" + batch_filename[idx] + ".json"), 'w', encoding='utf-8') as f:
                        json.dump(json_data, f,
                                  ensure_ascii=False, indent="\t")
            return fake_data


# min-max Normalization (최소-최대 정규화). 모든 feature에 대해 각각의 최소값 0, 최대값 1로, 나머지값은 그 사이에 되게끔 변경
def normal(array, min_val, max_val):
    return (array-min_val)/(max_val-min_val)


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)

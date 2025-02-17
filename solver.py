import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from readfiles import See_loss

import torch
import torch.nn as nn
import torch.optim as optim

from readfiles import printProgressBar
from networks import SEDCNN_WAXD
from measure import compute_measure,compute_measure_simple
from skimage import io


class Solver(object):
    def __init__(self,args,data_loader):
        self.mode=args.mode
        self.data_loader=data_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_epochs = args.decay_epochs
        self.save_epochs = args.save_epochs
        self.test_epochs= args.test_epochs


        self.SEDCNN_WAXD = SEDCNN_WAXD()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.SEDCNN_WAXD = nn.DataParallel(self.SEDCNN_WAXD)
        self.SEDCNN_WAXD.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.SEDCNN_WAXD.parameters(), self.lr)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'SEDCNN4_WAXD_{}epochs.ckpt'.format(iter_))
        torch.save(self.SEDCNN_WAXD.state_dict(), f)

    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'SEDCNN4_WAXD_{}epochs.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f).items():
                n = k[7:]
                state_d[n] = v
            self.SEDCNN_WAXD.load_state_dict(state_d)
        else:
            self.SEDCNN_WAXD.load_state_dict(torch.load(f))

    def warmup_cosine(self, current_epoch, lr_min=0, lr_max=0.1, warmup_epoch = 10):

        lr_max = self.lr
        max_epoch = self.num_epochs
        warmup_epoch = self.decay_epochs
        
        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        else:
            lr = lr_min + (lr_max-lr_min)*(1 + np.cos(np.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    
    def save_pre(self,pred,fig_name):

        fig_path = os.path.join(self.save_path, 'Revised_tif')

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
            
        pred=pred.numpy()
        io.imsave(os.path.join(fig_path , 'Revised_{}.tif'.format(fig_name)),np.float32(pred))
    
    def train(self):
        train_losses=[]
        total_iters=0
        start_time=time.time()
        for epoch in range(1, self.num_epochs+1):
            self.SEDCNN_WAXD.train(True)
            epoch_loss = 0.0
            for iter_, (x,y) in enumerate(self.data_loader['train']):
                total_iters += 1

                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)

                pred = self.SEDCNN_WAXD(x)
                loss = self.criterion(pred, y) 
                self.SEDCNN_WAXD.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                epoch_loss += loss.item()

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader['train']), loss.item(), 
                                                                                                        time.time() - start_time))
                    
            # learning rate decay
            self.warmup_cosine(epoch)

                # 每 test_epochs 进行验证
            if epoch % int(self.save_epochs) == 0:  
                self.save_model(epoch)


    def test(self):
        
        # 删除当前对象中的SEDCNN_WAXD属性
        del self.SEDCNN_WAXD
        #load
        self.SEDCNN_WAXD = SEDCNN_WAXD().to(self.device)
        self.load_model(self.test_epochs)

        with torch.no_grad():
            for i, (x) in enumerate(self.data_loader['test']):
                shape_1 = x.shape[-2]
                shape_2 = x.shape[-1]

                #print(x.shape)
                x = x.float().to(self.device)
                input = x.permute(1, 0, 2, 3).contiguous()
                #print(input.shape)
                pred = self.SEDCNN_WAXD(input)
                pred = torch.mean(pred, dim=0)
                #print(pred.shape)

                # denormalize, truncate
                pred = pred.view(shape_1, shape_2).cpu().detach()*300

                self.save_pre(pred,i)
            
            print('\n')
            print('Test finished!')
            
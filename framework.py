import torch
# from torch._C import long
import torch.nn as nn
from torch.autograd import Variable as V
from discriminator import FCDiscriminator
from dis_loss import bce_loss
from mse_loss import mse_loss
from scipy.ndimage.morphology import *

import cv2
import numpy as np
from skimage.morphology import skeletonize

class MyFrame_Src():
    def __init__(self, net, loss, lr=2e-4, mode='test', evalmode = False):
        self.net = net().cuda()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        self.mode = mode
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, ske_mask = None,  t_img_batch=None, t_mask=None,t_ske_mask=None, img_id=None):  #  t_img_batch, dist_mask=None,  t_img_batch, t_ske_mask=None, ske_mask = None,
        self.img = img_batch
        self.t_img = t_img_batch
        # print(self.img)
        self.mask = mask_batch
        self.t_mask = t_mask
        self.t_ske_mask = t_ske_mask
        self.ske_mask = ske_mask
       

    def set_test_input(self, img_batch, mask_batch, img_id=None): #
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img)   

        return mask  

        
    def forward(self, volatile=False):
        if self.mode=='test':
            self.img = V(self.img.cuda())
            if self.mask is not None:
                self.mask = V(self.mask.cuda())
        else:
            self.img = V(self.img.cuda())
            # self.img1 = V(self.t_img.cuda())
            if self.mask is not None:
                self.mask = V(self.mask.cuda())
                # self.mask1 = V(self.t_mask.cuda())
                # self.t_ske_mask = V(self.t_ske_mask.cuda())
                self.ske_mask = V(self.ske_mask.cuda())

    def forward_t(self, volatile=False):
        self.img = V(self.img.cuda())
        # self.img1 = V(self.t_img.cuda())
        if self.mask is not None:
            self.mask = V(self.mask.cuda())
            # self.mask1 = V(self.t_mask.cuda())
            # self.t_ske_mask = V(self.t_ske_mask.cuda())
            self.ske_mask = V(self.ske_mask.cuda())


    def optimize(self):

        self.optimizer.zero_grad()

        self.forward()
        

        # pred, pred1, e4= self.net.forward(self.img)
        pred = self.net.forward(self.img)

        loss_r  = self.loss(self.mask, pred) 
        # loss_sk = self.loss(self.ske_mask, pred1) 
        # loss_t = loss_r + loss_sk
        loss_t = loss_r
        if not torch.isnan(loss_t):
            loss_t.backward()
        else:
            print('here')

        

        
        
        self.optimizer.step()

        lost_total = loss_r.item() #+loss_sk.item()
        
        return lost_total
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def src_load(self, path):
        self.net.load_state_dict(torch.load(path), strict= False)
        # self.net.load_state_dict(torch.load(path), strict= True)
        
    
    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # print (mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, mode='test', evalmode = False):
        self.net = net().cuda()
        self.model_D1 = FCDiscriminator(num_classes=512)

        self.model_D1.train()
        self.model_D1.cuda()
        # self.net = nn.parallel.DistributedDataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.optimizer_D1 = torch.optim.Adam(self.model_D1.parameters(), lr=1e-4, betas=(0.9, 0.99))
        
        self.loss = loss()
        self.loss1 = bce_loss()
        self.loss2 = mse_loss()
        
        self.old_lr = lr
        self.mode = mode
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, ske_mask = None,  t_img_batch=None, t_mask=None,t_ske_mask=None, img_id=None):  #  t_img_batch, dist_mask=None,  t_img_batch, t_ske_mask=None, ske_mask = None,
        self.img = img_batch
        self.t_img = t_img_batch
        # print(self.img)
        self.mask = mask_batch
        self.t_mask = t_mask
        self.t_ske_mask = t_ske_mask
        self.ske_mask = ske_mask
        

    def set_test_input(self, img_batch, mask_batch, img_id=None): #
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
  
    
    
    def test_batch(self):
        self.forward(volatile=True)
        
        mask =  self.net.forward(self.img)    #.cpu().data.numpy().squeeze(1)
        
        return mask  #, self.img_id

        
    def forward(self, volatile=False):
        if self.mode=='test':
            self.img = V(self.img.cuda())
            if self.mask is not None:
                self.mask = V(self.mask.cuda())
        else:
            self.img = V(self.img.cuda())
            self.img1 = V(self.t_img.cuda())
            if self.mask is not None:
                self.mask = V(self.mask.cuda())
                self.mask1 = V(self.t_mask.cuda())
                self.t_ske_mask = V(self.t_ske_mask.cuda())
                self.ske_mask = V(self.ske_mask.cuda())

    def forward_t(self, volatile=False):
        self.img = V(self.img.cuda())
        self.img1 = V(self.t_img.cuda())
        if self.mask is not None:
            self.mask = V(self.mask.cuda())
            self.mask1 = V(self.t_mask.cuda())
            self.t_ske_mask = V(self.t_ske_mask.cuda())
            self.ske_mask = V(self.ske_mask.cuda())



    def optimize(self):

        ct_loss_ = True
        # ct_loss_ = False

        adv_loss_ = True
        # adv_loss_ = False

        self.optimizer.zero_grad()

        self.forward_t()
        
        self.img_c = torch.cat((self.img, self.img1))
        self.mask_c = torch.cat((self.mask, self.mask1))
        self.ske_mask_c = torch.cat((self.ske_mask, self.t_ske_mask))

        pred, pred1, e4= self.net.forward(self.img_c)

        loss_r  = self.loss(self.mask_c, pred) 
        loss_sk = self.loss(self.ske_mask_c, pred1) 

        loss_r = loss_r + loss_sk 
        if not torch.isnan(loss_r):
            loss_r.backward(retain_graph=True)
        else:
            print('here')

        
        if ct_loss_:
            pred1 = torch.sigmoid(pred1)
            pred = torch.sigmoid(pred)


            # pred = pred.detach()
            # pred2 = pred2.detach()

            loss_ct = 0.001*self.loss2(pred1[self.ske_mask_c==1], pred[self.ske_mask_c==1]) + torch.tensor(1e-12)

            if not (torch.isnan(loss_ct) or loss_ct == torch.tensor(1e-12)):
                if adv_loss_:
                    loss_ct.backward(retain_graph=True)
                else:
                    loss_ct.backward()
            else:
                print('here')
            # kl_loss.backward()   

        
        if adv_loss_:

            
            self.optimizer_D1.zero_grad()
            for param in self.model_D1.parameters():
                param.requires_grad = False

            source_label = 0
            target_label = 1

            mid = int(e4.shape[0]/2)
            D_out1 = self.model_D1(e4[mid:,:,:,:])
            loss_adv_target1 = self.loss1(D_out1, 
                                        V(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
                                        
            loss_adv_target1 = 0.01 * loss_adv_target1

            loss_adv_target1.backward()      

            self.optimizer.step()

            for param in self.model_D1.parameters():
                    param.requires_grad = True

            # train D with source
            f_pred1 = e4.detach()
            D_out1_1 = self.model_D1(f_pred1[:mid,:,:,:])

            loss_D1 = self.loss1(D_out1_1,
                                V(torch.FloatTensor(D_out1_1.data.size()).fill_(source_label)).cuda())


            # loss_D1 = loss_D1 / 2
            loss_D1.backward(retain_graph=True)

            # loss_D_value1 = loss_D_value1 + loss_D1.data.cpu().numpy()

            # train D with target
            pred_target1 = f_pred1[mid:,:,:,:]
            D_out1_2 = self.model_D1.forward(pred_target1)

            loss_D2 = self.loss1(D_out1_2,
                                V(torch.FloatTensor(D_out1_2.data.size()).fill_(target_label)).cuda())



            loss_D2.backward()
            
            self.optimizer_D1.step()
        else:
            self.optimizer.step()

        lost_total = loss_r.item()
        return lost_total, loss_adv_target1.item(), loss_ct.item()

        
    def save(self, path, path1):
        torch.save(self.net.state_dict(), path)
        torch.save(self.model_D1.state_dict(), path1)
    def src_load(self, path):
        # self.net.load_state_dict(torch.load(path), strict= False)
        self.net.load_state_dict(torch.load(path), strict= True)
        
    def load(self, path, path1):
        self.net.load_state_dict(torch.load(path), strict= False)
        self.model_D1.load_state_dict(torch.load(path1), strict= False)
        # for param in self.net.features.parameters():
        #     print(param)
        #     param.requires_grad = False
    
    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # print (mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr


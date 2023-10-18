import torch
# from torch._C import long
import torch.nn as nn
from torch.autograd import Variable as V
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
        # lost_total = loss_sk.item()
        # return lost_total, loss_adv_target1.item()#, kl_loss.item()
        # return lost_total, loss_ct.item()#, kl_loss.item()
        # return lost_total, loss_adv_target1.item(), loss_ct.item()
        # return lost_total, 0 ,loss_ct.item()
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

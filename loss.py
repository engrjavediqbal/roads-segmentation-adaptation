from colorsys import rgb_to_hls
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

import cv2
import numpy as np
class dice_bce_loss(nn.Module):
    def __init__(self,batch=True, ignore_label = 50):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.ignore_label = ignore_label
        self.bce_loss = nn.BCEWithLogitsLoss() #reduction='none'
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            # i = torch.sum(y_true)
            # j = torch.sum(y_pred)
            # intersection = torch.sum(y_true * y_pred)
            intersection = y_true * y_pred
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (y_true + y_pred + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score #score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_true, y_pred):
        assert not y_true.requires_grad
        assert y_pred.dim() == 4
        assert y_true.dim() == 4
        # assert y_pred.size(2) == y_true.size(1), "{0} vs {1} ".format(y_pred.size(2), y_true.size(1))
        # assert y_pred.size(3) == y_true.size(2), "{0} vs {1} ".format(y_pred.size(3), y_true.size(3))
        # n, c, h, w = y_pred.size()
        # target_mask = (y_true >= 0) * (y_true != self.ignore_label)
        # row, col = np.where(y_true == self.ignore_label) 
        # y_true = y_true[target_mask]
        # if not y_true.data.dim():
        #     return V(torch.zeros(1))
        # y_pred = y_pred.transpose(1, 2).transpose(2, 3).contiguous()
        # y_pred = y_pred[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(n, c, h, w)
        # y_true = y_true.view(n, c, h, w)
        # print(y_true.unique())
        # loss = F.cross_entropy(y_pred, y_true, ignore_index=self.ignore_label)
        # val = torch.ones(y_true.size()).cuda()
        # val1 = torch.zeros(y_true.size()).cuda()
        # weights = torch.where(y_true != self.ignore_label, val, val1)
        # weights[row, col] = 0
        # bce_loss = nn.BCEWithLogitsLoss(weight = weights)   
        bce_loss = nn.BCEWithLogitsLoss()
        # a =  bce_loss(y_pred, y_true)
        a = bce_loss(y_pred[y_true!=50], y_true[y_true!=50])
        # b =  self.soft_dice_loss(y_true, y_pred)
        return a

class fl_bce_loss(nn.Module):
    def __init__(self,batch=True, ignore_label = 50):
        super(fl_bce_loss, self).__init__()
        self.batch = batch
        self.ignore_label = ignore_label
        self.bce_loss = nn.BCEWithLogitsLoss() #reduction='none'
        # self.sig = torch.sigmoid()
        self.gamma = 2.0
        self.alpha = 0.5
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            # i = torch.sum(y_true)
            # j = torch.sum(y_pred)
            # intersection = torch.sum(y_true * y_pred)
            intersection = y_true * y_pred
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (y_true + y_pred + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score #score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_true, y_pred):
        assert not y_true.requires_grad
        assert y_pred.dim() == 4
        assert y_true.dim() == 4
        eps = 1e-12
        prob = torch.sigmoid(y_pred)
        log_prob = torch.log(prob+eps)
        log_prob_ = torch.log(1-prob+eps)
        mask = y_true!=50
        F_loss = ((1-self.alpha) * y_true[mask] * (1-prob[mask])**self.gamma * log_prob[mask]) + (self.alpha * (1-y_true[mask]) * prob[mask]**self.gamma * log_prob_[mask])
        # F_loss = (y_true[mask] * ((1-prob[mask])**self.gamma) * log_prob[mask]) + ((1-y_true[mask]) * (prob[mask]**self.gamma) * log_prob_[mask])
        F_loss = -F_loss.mean()

        two_losses = True
        if two_losses:
            bce_loss = nn.BCEWithLogitsLoss()
            ce_loss = bce_loss(y_pred[mask], y_true[mask])
            F_loss += ce_loss

        
        return F_loss
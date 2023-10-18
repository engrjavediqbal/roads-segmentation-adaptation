import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

import cv2
import numpy as np

class mse_loss(nn.Module):
    def __init__(self,batch=True):
        super(mse_loss, self).__init__()
        self.batch = batch
        self.mse_loss = nn.MSELoss() #reduction='none'
        
    def __call__(self, y_pred, y_true):  
        if y_true.sum()==0:
            a = 0.0
        else:
            a =  self.mse_loss(y_pred, y_true)
        # a =  self.bce_loss(y_true, y_pred)
        # b =  self.soft_dice_loss(y_true, y_pred)
        return a
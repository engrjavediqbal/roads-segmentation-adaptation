import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

import cv2
import numpy as np

class bce_loss(nn.Module):
    def __init__(self,batch=True):
        super(bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCEWithLogitsLoss() #reduction='none'
        
    def __call__(self, y_pred, y_true):  
        a =  self.bce_loss(y_pred, y_true)
        # a =  self.bce_loss(y_true, y_pred)
        # b =  self.soft_dice_loss(y_true, y_pred)
        return a
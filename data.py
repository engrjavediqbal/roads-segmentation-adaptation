"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os
from scipy.ndimage.morphology import *
from skimage.morphology import skeletonize

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask, gt_dt,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        gt_dt = cv2.warpPerspective(gt_dt, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
    return image, mask, gt_dt

def randomHorizontalFlip(image, mask, gt_dt, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        gt_dt = cv2.flip(gt_dt, 1)

    return image, mask, gt_dt

def randomVerticleFlip(image, mask, gt_dt, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        gt_dt = cv2.flip(gt_dt, 0)

    return image, mask, gt_dt

def randomRotate90(image, mask, gt_dt, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
        gt_dt=np.rot90(gt_dt)

    return image, mask, gt_dt

def centerCrop(img, mask, output_size):
    h, w = img.shape[0:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    mask =mask[i:i + th, j:j + tw]
    image = img[i:i + th, j:j + tw, :]
    return image, mask 

def randomCrop(image, gt, gt_dt, size):
    w, h, _ = image.shape
    crop_h, crop_w = size

    start_x = np.random.randint(0, w - crop_w)
    start_y = np.random.randint(0, h - crop_h)

    image = image[start_x : start_x + crop_w, start_y : start_y + crop_h, :]
    gt = gt[start_x : start_x + crop_w, start_y : start_y + crop_h]
    gt_dt = gt_dt[start_x : start_x + crop_w, start_y : start_y + crop_h]
    return image, gt, gt_dt

def default_loader(id, root, split):
    if split == 'train':
      
        im_path = os.path.join(root,'images/{}.png').format(id)
        m_path = os.path.join(root,'gt/{}.png').format(id)
        ske_gt = os.path.join(root,'ske_gt/{}.png').format(id)
    
        img = cv2.imread(im_path)
        if(img is None):
            print(im_path)
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        d_mask = cv2.imread(ske_gt, cv2.IMREAD_GRAYSCALE)

        d_mask[d_mask==255] = 1
        # kernel = np.ones((3,3),np.uint8)
        # d_mask = cv2.dilate(d_mask.copy(),kernel,iterations = 1)


        # img, mask, d_mask = randomCrop(img, mask, d_mask, (1024, 1024))

        img = cv2.resize(img, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
        d_mask = cv2.resize(d_mask, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)

        siz = (768, 768)
        # siz = (640, 640)
        # siz = (512, 512)

        img, mask, d_mask = randomCrop(img, mask, d_mask, siz)


      # img = cv2.imread(os.path.join(root,'{}_sat.jpg').format(id))
      # mask = cv2.imread(os.path.join(root,'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
      # img = randomHueSaturationValue(img,
      #                               hue_shift_limit=(-30, 30),
      #                               sat_shift_limit=(-5, 5),
      #                               val_shift_limit=(-15, 15))

        img, mask, d_mask = randomShiftScaleRotate(img, mask, d_mask,
                                            shift_limit=(-0.1, 0.1),
                                            scale_limit=(-0.1, 0.1),
                                            aspect_limit=(-0.1, 0.1),
                                            rotate_limit=(-0, 0))
        img, mask, d_mask = randomHorizontalFlip(img, mask, d_mask)
        img, mask, d_mask = randomVerticleFlip(img, mask, d_mask)
        img, mask, d_mask = randomRotate90(img, mask, d_mask)

        mask = np.expand_dims(mask, axis=2)
        d_mask = np.expand_dims(d_mask, axis=2)
        d_mask = np.array(d_mask, np.float32).transpose(2,0,1)
    #   mask = np.array(mask, np.float32).transpose(2,0,1)
    #   print(d_mask.shape)
    #   print(mask.shape)
        img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    #   mask[mask==255] = 1
    #   d_mask[d_mask==255] = 1
        mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
        mask[mask>=0.76] = 1
        mask[mask<=0.76] = 0

        skeleton = skeletonize(mask[0])
        d_mask = (1*skeleton).astype(np.uint8)
        d_mask = np.expand_dims(d_mask, axis=2)
        d_mask = np.array(d_mask, np.float32).transpose(2,0,1)
    
    return img, mask.copy(), d_mask.copy()
    
class ImageFolder_Src(data.Dataset):

    def __init__(self, trainlist, root, split):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root
        self.split = split

    def __getitem__(self, index):
        id = self.ids[index]
        # try:
        img, mask, d_mask = self.loader(id, self.root, self.split)
        # print(img)
        # if (img is not None and mask is not None):
        # d_mask = d_mask.copy()
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        d_mask = torch.Tensor(d_mask)
        # except:
        #     return None
    
            # print(img.shape)
            # print(mask.shape)
        return img, mask, d_mask

    def __len__(self):
        return len(self.ids)
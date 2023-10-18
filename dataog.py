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
torch.manual_seed(0)

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

def randomShiftScaleRotate(image, mask,
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

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def centerCrop(img, mask, output_size):
    h, w = img.shape[0:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    mask =mask[i:i + th, j:j + tw]
    image = img[i:i + th, j:j + tw, :]
    return image, mask 

def randomCrop(image, gt, size):
    w, h, _ = image.shape
    crop_h, crop_w = size

    start_x = np.random.randint(0, w - crop_w)
    start_y = np.random.randint(0, h - crop_h)

    image = image[start_x : start_x + crop_w, start_y : start_y + crop_h, :]
    gt = gt[start_x : start_x + crop_w, start_y : start_y + crop_h]

    return image, gt

def default_loader(id, root, split):
    if split == 'train':
      
        im_path = os.path.join(root,'images/{}.png').format(id)
        m_path = os.path.join(root,'gt/{}.png').format(id)
        
      #   print(m_path)
        img = cv2.imread(im_path)
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
  
      # img = cv2.imread(os.path.join(root,'{}_sat.jpg').format(id))
      # mask = cv2.imread(os.path.join(root,'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
      # img = randomHueSaturationValue(img,
      #                               hue_shift_limit=(-30, 30),
      #                               sat_shift_limit=(-5, 5),
      #                               val_shift_limit=(-15, 15))
        img, mask = randomCrop(img, mask, (1024, 1024))
    #   img = cv2.resize(img, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
    #   mask = cv2.resize(mask, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)

        img, mask = randomShiftScaleRotate(img, mask,
                                          shift_limit=(-0.1, 0.1),
                                          scale_limit=(-0.1, 0.1),
                                          aspect_limit=(-0.1, 0.1),
                                          rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)
    #   print(mask.shape)
        mask = np.expand_dims(mask, axis=2)
      
      #   print(d_mask.shape)
        img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
        mask[mask>=0.76] = 1
        mask[mask<=0.76] = 0

    elif split == 'test_src':
      
        im_path = os.path.join(root,'images/{}.png').format(id)
        m_path = os.path.join(root,'gt/{}.png').format(id)
        
      #   print(m_path)
        img = cv2.imread(im_path)
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
  
      # img = cv2.imread(os.path.join(root,'{}_sat.jpg').format(id))
      # mask = cv2.imread(os.path.join(root,'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
      # img = randomHueSaturationValue(img,
      #                               hue_shift_limit=(-30, 30),
      #                               sat_shift_limit=(-5, 5),
      #                               val_shift_limit=(-15, 15))
        dim = (1280,1280)
        # img, mask = randomCrop(img, mask, (1024, 1024))
        # img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)

        # img = cv2.resize(img, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)

        # img, mask = centerCrop(img, mask, (768, 768))
        img, mask = centerCrop(img, mask, (1024, 1024))

        # img, mask = randomShiftScaleRotate(img, mask,
        #                                   shift_limit=(-0.1, 0.1),
        #                                   scale_limit=(-0.1, 0.1),
        #                                   aspect_limit=(-0.1, 0.1),
        #                                   rotate_limit=(-0, 0))
        # img, mask = randomHorizontalFlip(img, mask)
        # img, mask = randomVerticleFlip(img, mask)
        # img, mask = randomRotate90(img, mask)
    #   print(mask.shape)
        mask = np.expand_dims(mask, axis=2)
      
      #   print(d_mask.shape)
        img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
        mask[mask>=0.76] = 1
        mask[mask<=0.76] = 0
    
    else:
        im_path = os.path.join(root,'images/{}_sat.jpg').format(id)
        m_path = os.path.join(root,'gt/{}_mask.png').format(id)

        # im_path = os.path.join(root,'images/{}.tiff').format(id)
        # m_path = os.path.join(root,'gt/{}.tif').format(id)

      # im_path = os.path.join(root,'self_train_ske_hys/ske_pseudo_lbl_r1{}_sat.jpg').format(id)
      # m_path = os.path.join(root,'{}_mask.png').format(id)
      # im_path = os.path.join(root,'images/{}.png').format(id)
      # m_path = os.path.join(root,'gt/{}.png').format(id)
    #   ske = os.path.join(root,'ske_gt/{}.png').format(id)
        img = cv2.imread(im_path)
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
    #   ske_gt = cv2.imread(ske, cv2.IMREAD_GRAYSCALE)
        # img, mask = centerCrop(img, mask, (1024, 1024))
        mask = np.expand_dims(mask, axis=2)
        size_ = (1024,1024)
        # size_ = (768, 768)
        # size_ = (1280,1280)
        
        # size_ = (1504,1504)
        # size_ = (1920, 1920)
        # size_ = (1760, 1760) # best for mDataset
        # size_ = (2080, 2080)
        img = cv2.resize(img, size_, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size_, interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=2)

        img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    #   mask = mask*255
    #   mask[mask==255] = 1
      # mask[mask>=0.76] = 1
      # mask[mask<=0.76] = 0
   
    #mask = abs(mask-1)
    return img, mask, id
    
class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root, split):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root
        self.split = split

    def __getitem__(self, index):
        # print(index)
        id = self.ids[index]
        # try:
        img, mask, im_id = self.loader(id, self.root, self.split)
        # print(img)
        # if (img is not None and mask is not None):
        
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        # ske_mask = torch.Tensor(ske_mask)
       
        # except:
        #     return None
    
            # print(img.shape)
            # print(mask.shape)
        return img, mask, im_id

    # def __iter__(self):
    #     return iter(range(len(self.ids)))

    def __len__(self):
        return len(self.ids)
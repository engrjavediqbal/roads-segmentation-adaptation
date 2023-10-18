import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from networks.dinknet import DinkNet34
from framework import MyFrame_Src
from loss import dice_bce_loss
from dataog import ImageFolder
import argparse

torch.manual_seed(0)
# plt1.use('Qt5Agg')


def main(args):


    BATCHSIZE_PER_CARD = 1

   
    solver = MyFrame_Src(DinkNet34, dice_bce_loss, 0.001, args.mode)

    model_path = args.ROOT + '/weights/' + args.model_name

    solver.src_load(model_path)
    print('testing model: ' + model_path)


    trainlist = [i_id.strip() for i_id in open(args.list_path)]
    dataset = ImageFolder(trainlist, args.data_dir_tar, split = args.mode)
    # print(len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    def per_class_iu(hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def fast_hist(a, b, n):
        # print(a)
        # print(b)
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

    hist = np.zeros((2, 2))    
    # hist1 = np.zeros((2, 2))  
    data_loader_iter = iter(data_loader)
    f1_ls = []
    prec = []
    rec = []

    s_f1_ls = []
    s_prec = []
    s_rec = []


    with torch.no_grad():
        for img, mask, im_id in data_loader_iter:   
            solver.set_test_input(img, mask)
            # pred, pred1, e4 = solver.test_batch()
            pred, _, _ = solver.test_batch()
            pred = torch.sigmoid(pred)

            pred[pred>=0.5] = 1
            pred[pred<0.5] = 0

            # cv2.imwrite(ROOT + path1+str(im_id[0])+'_mask.png', 255*pred[0,0,:,:].detach().cpu().numpy())
            
            correctLabel = mask.view(-1, args.image_size, args.image_size).long()
            
            hist += fast_hist(
            pred.view(pred.size(0), -1).detach().cpu().numpy(),
            correctLabel.view(correctLabel.size(0), -1).cpu().numpy(), 2)


    mean_iou = per_class_iu(hist)
    print("MIOU:  ", 100 * np.nanmean(mean_iou), "Class_IOU:  ", 100 * mean_iou[1])

    sumH = np.sum(hist, axis=0)
    sumV = np.sum(hist, axis=1)
    prec = hist[1,1]/sumV[1]
    rec = hist[1,1]/sumH[1]
    f1_ = 2*prec*rec / (prec+rec)

    print("F1_Score: ", np.mean(f1_))
    print("Precision: ", np.mean(prec))
    print("Recall: ", np.mean(rec))


if __name__=='__main__':

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT', default=ROOT_DIR, help='Model for upsampling')
    parser.add_argument('--data_dir_tar', default=ROOT_DIR+'/data/deepGlobe/', help='train or test') 
    parser.add_argument('--mode', default='test', help='train or test')
    parser.add_argument('--gpu', default='0', help='which gpu to use')
    parser.add_argument('--model_name', default='SkDecoder_FSA_r1_3.pth', help='Upsampling Ratio')  #
    parser.add_argument('--list_path', default=ROOT_DIR+'/data/deepGlobe/val.txt', help='Point Number')
    parser.add_argument('--image_size', type=int, default=1024, help='Batch Size during training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training')
    args = parser.parse_args()

    

    main(args)
from math import nan
import torch
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from time import time

from networks.dinknet import DinkNet34
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder_Src
from data_copy import ImageFolder_Target
from target_data import Tar_ImageFolder
from generate_pLabels import assign_labels
import re
import argparse


torch.autograd.detect_anomaly()
# torch.manual_seed(0)
# train_epoch_loss = nan
# if train_epoch_loss is nan:
#     print('Nannnn')

_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""
 
np_str_obj_array_pattern = re.compile(r'[SaUO]')


# i = cv2.imreda()
 
error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

def generate_labels(args, solver, round, round_folder):

    print('Generating labels using')

    batchsize = 1

    trainlist = [i_id.strip() for i_id in open(args.list_path_tar)]

    dataset = Tar_ImageFolder(trainlist, args.data_dir_tar, split = 'test')

    data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True)
    
    data_loader_iter = iter(data_loader)

    rd_folder = round_folder + '/gt'
    sk_folder = round_folder + '/gt_sk'

    if not os.path.exists(rd_folder):
            os.makedirs(rd_folder)
    if not os.path.exists(sk_folder):
            os.makedirs(sk_folder)

    im_id = 0
    with torch.no_grad():
        for img, mask, img_id in data_loader_iter:   
            solver.set_test_input(img, mask)
            pred, pred1, _= solver.test_batch()
            pred1 = torch.sigmoid(pred1)
            pred = torch.sigmoid(pred)
            
            size_ = (args.image_size, args.image_size)
            pred = pred.view(size_[0], size_[1]).detach().cpu().numpy()
            pred1 = pred1.view(size_[0], size_[1]).detach().cpu().numpy()

            pl_road, pl_ske = assign_labels(pred, pred1, round)

            cv2.imwrite(rd_folder + '/'+str(img_id[0])+'_mask.png', pl_road)
            cv2.imwrite(sk_folder + '/'+str(img_id[0])+'_mask.png', pl_ske)

            if im_id%50==0:
                print('Processed image: ', im_id)

            im_id +=1
        
    print('Pseudo-Labels Generation Completed')
            

def adapt_model(args, solver, round, round_folder, mylogs):

    print('Adapting models')

    mylog = open(mylogs,"a")

    batchsize = args.batch_size

    trainlist_src = [i_id.strip() for i_id in open(args.list_path_src)]
    trainlist_tar = [i_id.strip() for i_id in open(args.list_path_tar)]

    dataset = ImageFolder_Src(trainlist_src, args.data_dir_src, split = 'train')
    tar_dataset = ImageFolder_Target(trainlist_tar, args.data_dir_tar, round_folder, split = 'train')

    data_loader_src = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True)

    data_loader_tar = torch.utils.data.DataLoader(
        tar_dataset,
        batch_size=batchsize,
        shuffle=True)

    train_epoch_best_loss = 100.

    for epoch in range(1, args.adapt_epochs + 1):
        data_loader_iter_src = iter(data_loader_src)
        data_loader_iter_tar = iter(data_loader_tar)
        train_epoch_loss = 0
        epoch_adv_loss = 0
        ct_epoch_loss = 0

        j = 0
        for img, mask, ske_mask in data_loader_iter_src:  #, ske_mask, t_ske_mask
            t_img, t_mask,t_ske_mask = next(data_loader_iter_tar)

            j += 1
            # if j%100 == 0:
            #     print(j)

            solver.set_input(img, mask, ske_mask, t_img, t_mask, t_ske_mask)  
            
            loss, adv_loss, ct_loss = solver.optimize()  #, kl_loss
            train_epoch_loss += loss
            epoch_adv_loss += adv_loss
            ct_epoch_loss += ct_loss

            if j%50 == 0:
                print("train Loss:   ", train_epoch_loss/j)
                print("adv Loss:   ", epoch_adv_loss/j)
                print("ct Loss:   ", ct_epoch_loss/j)
                
        train_epoch_loss /= len(data_loader_iter_src)
        epoch_adv_loss /= len(data_loader_iter_src)
        ct_epoch_loss /= len(data_loader_iter_src)

        print ('********')
        print ('epoch:',epoch)
        print ('train_loss: ',train_epoch_loss)
        # print ('adv_loss: ',epoch_adv_loss)
        # print ('SHAPE:',SHAPE)
        
        if train_epoch_loss is nan:
            break

        mpat = round_folder+'/'+args.model_name+'_'+str(round)+'_'+str(epoch)

        if (train_epoch_loss >= train_epoch_best_loss):
            no_optim += 1
            train_epoch_best_loss = train_epoch_loss
            e = epoch
            solver.save(mpat+'.pth', mpat+'_dis.pth')
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            e = epoch
            solver.save(mpat+'.pth', mpat+'_dis.pth')
        if no_optim >= 6:
            # print (mylog, 'early stop at %d epoch' % epoch)
            print ('early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.save(mpat+'.pth', mpat+'_dis.pth')
            solver.update_lr(5.0, factor = True)
    print ('Finish!')
    # mylog.close()

    return solver


def main(args):

    solver = MyFrame(DinkNet34, dice_bce_loss, 0.0002)
    solver.src_load(args.src_model)

    exp_folder = args.ROOT + '/experiments/'  + args.adapt_exp

    mylogs = exp_folder+"/logs.txt"

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)


    for round in range(args.adapt_rounds):

        round_folder = exp_folder + '/round_'  + str(round)

        if not os.path.exists(round_folder):
            os.makedirs(round_folder)
        
        print('Generate Pseudo_labels')
        generate_labels(args, solver, round, round_folder)

        print('Perform Adaptation')
        solver = adapt_model(args, solver, round, round_folder, mylogs)






if __name__=='__main__':

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT', default=ROOT_DIR, help='root folder path')
    parser.add_argument('--data_dir_src', default=ROOT_DIR+'/data/spaceNet/', help='Source dataset')
    parser.add_argument('--data_dir_tar', default=ROOT_DIR+'/data/deepGlobe/', help='Target dataset') 
    parser.add_argument('--mode', default='train', help='train or test')
    parser.add_argument('--gpu', default='0', help='which gpu to use')
    parser.add_argument('--model_name', default='spaceNet_to_deepGlobe_', help='model name to save')  #
    parser.add_argument('--list_path_src', default=ROOT_DIR+'/data/spaceNet/train.txt', help='Source train list')
    parser.add_argument('--list_path_tar', default=ROOT_DIR+'/data/deepGlobe/train.txt', help='Target train list')
    parser.add_argument('--image_size', type=int, default=1024, help='Image Size during training')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training')
    parser.add_argument('--adapt_rounds', type=int, default=2, help='adaptation rounds during training')
    parser.add_argument('--adapt_epochs', type=int, default=3, help='adaptation epochs per round during training')
    parser.add_argument('--src_model', default=ROOT_DIR+'/weights/src_spaceNet.pth', help='Source trained model')
    parser.add_argument('--adapt_exp', default='spaceNet2ddepGlobe_all', help='Source trained model')
    args = parser.parse_args()

    

    main(args)
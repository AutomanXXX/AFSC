import torch
import time
import datetime
from data_loader_self_cutpaste import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
from torch import optim
from model import DiscriminativeSubNetwork
from loss import SSIM
import os
import sys
import numpy as np
import random
import imgaug as ia
import cv2
from msgms import MSGMSLoss
import torch.nn.functional as F
from generic_util import trapezoid
from rgb2lab import lab_loss_metric

def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)

def ColorDifference(imgo,imgr):
    imglabo=cv2.cvtColor(imgo,cv2.COLOR_BGR2LAB)
    imglabr=cv2.cvtColor(imgr,cv2.COLOR_BGR2LAB)
    diff=(imglabr-imglabo)*(imglabr-imglabo)
    RD=diff[:,:,1]
    BD=diff[:,:,2]
    Result=RD+BD
    Result=cv2.blur(Result,(11,11))*0.001
    return Result

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(obj_names, args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)


    run_name_top = args.name + '_lr' + str(args.lr) + '_epochs' + str(args.epochs) + '_bs' + str(
        args.batch_size)

    if not os.path.exists(os.path.join(args.checkpoint_path, run_name_top + "/")):
        os.makedirs(os.path.join(args.checkpoint_path, run_name_top + "/"))

    for obj_name in obj_names:

        run_name = run_name_top + '/' + obj_name

        model = DiscriminativeSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)

        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2,
                                                   last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        msgms = MSGMSLoss().cuda()

        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/",
                                         resize_shape=[256, 256])

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)

        prev_time = time.time()


        for epoch in range(args.epochs):
            l2_loss_of_epoch = 0
            ssim_loss_of_epoch = 0
            loss_of_epoch = 0

            for i_batch, sample_batched in enumerate(dataloader):

                model.train()
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()

                gray_rec, last_tensor_mask1, last_tensor_mask2, last_tensor_mask3,_ = model(aug_gray_batch, len(dataloader), i_batch, epoch, args.fix_epoch, run_name)
                gray_rec = torch.sigmoid(gray_rec)

                lab_loss = lab_loss_metric(gray_rec, gray_batch)
                mgsgms_loss=msgms(gray_rec, gray_batch, as_loss=True)
                l2_loss = loss_l2(gray_rec, gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)

                regularization_loss = torch.sum(torch.abs(last_tensor_mask1)) + torch.sum(
                    torch.abs(last_tensor_mask2)) + torch.sum(torch.abs(last_tensor_mask3))
                

                if (epoch+1) < args.fix_epoch-5:
                    loss = l2_loss + ssim_loss + 0.000001 * regularization_loss + mgsgms_loss + 0.001*lab_loss
                else:
                    loss = l2_loss + ssim_loss + mgsgms_loss + 0.001*lab_loss  

                l2_loss_of_epoch += l2_loss.item()
                ssim_loss_of_epoch += ssim_loss.item()
                loss_of_epoch += loss.item()


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                batches_done = epoch * len(dataloader) + i_batch
                batches_left = args.epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(

                    "\r[%s][Epoch %d/%d][Batch %d/%d][l2_loss: %f, ssim_loss: %f][loss: %f] ETA: %s"
                    % (
                        obj_name,
                        epoch + 1,
                        args.epochs,
                        i_batch,
                        len(dataloader),
                        l2_loss.item(),
                        ssim_loss.item(),
                        loss.item(),
                        time_left
                    )
                )

            print(
                "\r[%s][Epoch %d/%d] [Batch %d/%d] [l2_loss_all: %f, ssim_loss_all: %f][loss_all: %f] ETA: %s"
                % (
                    obj_name,
                    epoch + 1,
                    args.epochs,
                    len(dataloader),
                    len(dataloader),
                    l2_loss_of_epoch,
                    ssim_loss_of_epoch,
                    loss_of_epoch,
                    time_left
                )
            )

            scheduler.step()

            if  (epoch + 1) == args.epochs:
                save_name = run_name + "_epoch" + str(epoch + 1)
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, save_name + ".pckl"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', action='store', type=float, default=0.0001)
    parser.add_argument('--epochs', action='store', type=int, default=800)
    parser.add_argument('--gpu_id', action='store', type=int, default=0)
    parser.add_argument('--data_path', action='store', type=str,
                        default=r'./mvtec_anomaly_detection/')
    parser.add_argument('--checkpoint_path', action='store', type=str, default='./checkpoint/')
    parser.add_argument('--fix_epoch', type=int, default=400)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--name', action='store', type=str, default='AFSC_6e_v1')


    args = parser.parse_args()

    obj_batch = [['carpet'],  # 0
                 ['grid'],     # 1
                 ['leather'],  # 2
                 ['tile'],     # 3
                 ['wood'],    # 4
                 ['pill'],     # 5
                 ['transistor'],  # 6
                 ['cable'],        # 7
                 ['zipper'],      # 8
                 ['toothbrush'],  # 9
                 ['metal_nut'],   # 10
                 ['hazelnut'],   # 11
                 ['screw'],      # 12
                 ['capsule'],    # 13
                 ['bottle'],    # 14
                 ]

    if int(args.obj_id) == -1:
        obj_list = [
                     'carpet',      #0
                     'grid',        #1
                     'leather',     #2
                     'tile',        #3
                     'wood',        #4

                     'pill',        #5
                     'transistor',  #6
                     'cable',       #7
                     'zipper',      #8
                     'toothbrush',  #9
                     'metal_nut',   #10
                     'hazelnut',    #11
                     'screw',       #12
                    'capsule',      #13
                     'bottle'       #14
                     ]

        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

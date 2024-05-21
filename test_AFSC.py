import torch
import torch.nn.functional as F
from data_loader_self_cutpaste import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, auc
from model import DiscriminativeSubNetwork
import os
import cv2
from loss import SSIM
from msgms import MSGMSLoss
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
from torchvision.transforms.functional import to_tensor
from skimage.metrics import structural_similarity as ssim
from skimage import measure
import pandas as pd
from statistics import mean as stat_mean


def mean_smoothing(amaps, kernel_size: int = 21):
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def ColorDifference(imgo, imgr):
    imglabo = cv2.cvtColor(imgo, cv2.COLOR_BGR2LAB)
    imglabr = cv2.cvtColor(imgr, cv2.COLOR_BGR2LAB)
    diff = (imglabr - imglabo) * (imglabr - imglabo)
    RD = diff[:, :, 1]
    BD = diff[:, :, 2]
    Result = RD + BD
    Result = cv2.blur(Result, (11, 11)) * 0.001
    return Result


def compute_pro(masks, amaps, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    # assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    # assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": stat_mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def test(obj_names, mvtec_path, checkpoint_path, run_name_top, which_epoch_model):
    obj_auroc_pixel_list = []
    obj_auroc_image_list = []
    obj_aupro_list = []
    for obj_name in obj_names:
        img_dim1 = 256
        img_dim2 = 256
        aupro_list = []
        run_name = run_name_top + '/' + obj_name
        save_name = run_name + "_epoch" + str(which_epoch_model)

        model = DiscriminativeSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, save_name + ".pckl"), map_location='cuda:0'),
                              strict=False)
        model.cuda()
        model.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim1, img_dim2])
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

        total_pixel_scores = np.zeros((img_dim1 * img_dim2 * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim1 * img_dim2 * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        pro_gt = []
        pro_out = []

        msgms = MSGMSLoss().cuda()

        for i_batch, sample_batched in enumerate(dataloader):

            with torch.no_grad():
                gray_batch = sample_batched["image"].cuda()
                is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
                anomaly_score_gt.append(is_normal)
                true_mask = sample_batched["mask"]

                true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

                gray_rec, _, _, _, network_input = model(gray_batch, 3, 1, 700, 400, run_name)
                gray_rec = torch.sigmoid(gray_rec)

                recimg = gray_rec.detach().cpu().numpy()[0]
                recimg = np.transpose(recimg, (1, 2, 0)) * 180
                recimg = recimg.astype('uint8')
                oriimg = gray_batch.detach().cpu().numpy()[0]
                oriimg = np.transpose(oriimg, (1, 2, 0)) * 180
                oriimg = oriimg.astype('uint8')
                # color
                colorD = ColorDifference(recimg, oriimg)

                # msgms
                mgsgmsmap = msgms(gray_rec, gray_batch, as_loss=False)
                mgsgmsmapmean = mean_smoothing(mgsgmsmap, 21)
                out_mask_gradient = mgsgmsmapmean.detach().cpu().numpy()

                out_mask_averaged = colorD[None, None, :, :] + out_mask_gradient  # (1, 1, img_dim1, img_dim2)

                image_score = np.max(out_mask_averaged)
                anomaly_score_prediction.append(image_score)
                flat_true_mask = true_mask_cv.flatten()
                flat_out_mask = out_mask_averaged.flatten()
                total_pixel_scores[mask_cnt * img_dim1 * img_dim2:(mask_cnt + 1) * img_dim1 * img_dim2] = flat_out_mask
                total_gt_pixel_scores[
                mask_cnt * img_dim1 * img_dim2:(mask_cnt + 1) * img_dim1 * img_dim2] = flat_true_mask
                mask_cnt += 1

                # for pro
                truegt = true_mask_cv[:, :, 0]
                outresult = out_mask_averaged[0, 0, :, :]
                pro_gt.append(truegt)
                pro_out.append(outresult)

                for i in range(out_mask_averaged.shape[0]):
                    if is_normal and true_mask[i].max() == 1:
                        aupro_list.append(compute_pro(true_mask[i].cpu().numpy().astype(int),
                                                      out_mask_averaged[i, 0, ...].squeeze()[np.newaxis, :, :]))

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim1 * img_dim2 * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim1 * img_dim2 * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)

        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_aupro_list.append(round(np.mean(aupro_list), 3))
        print(obj_name)
        print('%.1f' % (auroc * 100.0))
        print('%.1f' % (auroc_pixel * 100.0))
        print(str(round(np.mean(aupro_list) * 100.0, 1)))

    print(run_name)
    print('%.1f' % (np.mean(obj_auroc_image_list) * 100.0))
    print('%.1f' % (np.mean(obj_auroc_pixel_list) * 100.0))
    print(str(round(np.mean(obj_aupro_list) * 100.0, 1)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, default=-1)  # obj
    parser.add_argument('--gpu_id', action='store', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--base_model_name', action='store', type=str,
                        default='AFSC_mvtec')
    parser.add_argument('--which_epoch_model', action='store', type=int, default=800)
    parser.add_argument('--data_path', action='store', type=str,
                        default=r'./mvtec_anomaly_detection/')
    parser.add_argument('--checkpoint_path', action='store', type=str, default='./checkpoint/')

    args = parser.parse_args()

    obj_batch = [['carpet'],  # 0
                 ['grid'],  # 1
                 ['leather'],  # 2
                 ['tile'],  # 3
                 ['wood'],  # 4
                 ['pill'],  # 5
                 ['transistor'],  # 6
                 ['cable'],  # 7
                 ['zipper'],  # 8
                 ['toothbrush'],  # 9
                 ['metal_nut'],  # 10
                 ['hazelnut'],  # 11
                 ['screw'],  # 12
                 ['capsule'],  # 13
                 ['bottle'],  # 14
                 ]

    if int(args.obj_id) == -1:

        obj_list = [
            'carpet',  # 0
            'grid',  # 1
            'leather',  # 2
            'tile',  # 3
            'wood',  # 4

            'pill',  # 5
            'transistor',  # 6
            'cable',  # 7
            'zipper',  # 8
            'toothbrush',  # 9
            'metal_nut',  # 10
            'hazelnut',  # 11
            'screw',  # 12
            'capsule',  # 13
            'bottle'  # 14
        ]

        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        test(picked_classes, args.data_path, args.checkpoint_path, args.base_model_name, args.which_epoch_model)

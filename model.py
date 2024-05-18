import torch
import torch.nn as nn
import os
import numpy as np
from torch.nn import init


class EFCAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return y.expand_as(x)


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.fft2(x, dim=(-2, -1))
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = torch.complex(ffted[:, :c, ...], ffted[:, c:, ...])
        ffted = torch.fft.ifft2(ffted, dim=(-2, -1))

        output = torch.real(ffted)

        return output


class SpectralTransformer(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransformer, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class EFCm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(EFCm, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.in_cg = in_cg
        self.in_cl = in_cl
        self.out_cg = out_cg
        self.out_cl = out_cl

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransformer
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)

        self.efc = EFCAttention(kernel_size=3)

    def forward(self, x):

        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        x_l = x[:, :self.in_cl, :, :]
        x_g = x[:, self.in_cl:, :, :]

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        out_xl = self.relu_l(out_xl)
        out_xg = self.relu_g(out_xg)
        out_last = torch.cat([out_xl, out_xg], dim=1)

        out = out_last * self.efc(out_last)

        return out



class DiscriminativeSubNetwork(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, base_channels=64, out_features=False):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
        self.out_features = out_features

        mult0 = torch.ones(1, 256, 256)  # [0, 1]
        mult0 = mult0/2
        
        self.slope1_1 = 13.0
        mult1 = - torch.log(1. / mult0 - 1.) / self.slope1_1
        self.mult1 = torch.nn.Parameter(mult1, requires_grad=True)
        self.slope1_2 = 32.0

        self.slope2_1 = 13.0
        mult2 = - torch.log(1. / mult0 - 1.) / self.slope2_1
        self.mult2 = torch.nn.Parameter(mult2, requires_grad=True)
        self.slope2_2 = 32.0

        self.slope3_1 = 13.0
        mult3 = - torch.log(1. / mult0 - 1.) / self.slope3_1
        self.mult3 = torch.nn.Parameter(mult3, requires_grad=True)
        self.slope3_2 = 32.0

        self.frequency_mask1 = torch.zeros(mult1.shape).cuda()
        self.frequency_mask2 = torch.zeros(mult2.shape).cuda()
        self.frequency_mask3 = torch.zeros(mult3.shape).cuda()

    def forward(self, x, len_dataloader, i_batch, epoch, fix_epoch, run_name):


        fft_im_ = torch.fft.fft2(x, dim=(-2, -1))

        last_tensor = torch.cat([torch.real(fft_im_), torch.imag(fft_im_)], dim=1)

        if (epoch+1) < fix_epoch:

            prob_mask_tensor1 = torch.sigmoid(self.slope1_1 * self.mult1)
            thresh_tensor1 = (0 * prob_mask_tensor1) + torch.rand(1, 256, 256).cuda()
            last_tensor_mask1 = torch.sigmoid(self.slope1_2 * (prob_mask_tensor1 - thresh_tensor1))


            prob_mask_tensor2 = torch.sigmoid(self.slope2_1 * self.mult2)
            thresh_tensor2 = (0 * prob_mask_tensor2) + torch.rand(1, 256, 256).cuda()
            last_tensor_mask2 = torch.sigmoid(self.slope2_2 * (prob_mask_tensor2 - thresh_tensor2))

            prob_mask_tensor3 = torch.sigmoid(self.slope3_1 * self.mult3)
            thresh_tensor3 = (0 * prob_mask_tensor3) + torch.rand(1, 256, 256).cuda()
            last_tensor_mask3 = torch.sigmoid(self.slope3_2 * (prob_mask_tensor3 - thresh_tensor3))

            if (epoch+1) >= fix_epoch-5:
                with torch.no_grad():
                    le1 = torch.less_equal(thresh_tensor1, prob_mask_tensor1)
                    le1 = le1.float()
                    le2 = torch.less_equal(thresh_tensor2, prob_mask_tensor2)
                    le2 = le2.float()
                    le3 = torch.less_equal(thresh_tensor3, prob_mask_tensor3)
                    le3 = le3.float()

                    self.frequency_mask1 += le1
                    self.frequency_mask2 += le2
                    self.frequency_mask3 += le3
                    if (epoch+1) == fix_epoch-1 and i_batch == len_dataloader - 1:
                        frequency_mask_flod_path = "./save_masks/" + run_name + '/'
                        if not os.path.exists(frequency_mask_flod_path):
                            os.makedirs(frequency_mask_flod_path)

                        last_tensor_mask1 = torch.where(self.frequency_mask1>(len_dataloader*5*5/10), 1, 0)
                        last_tensor_mask2 = torch.where(self.frequency_mask2>(len_dataloader*5*5/10), 1, 0)
                        last_tensor_mask3 = torch.where(self.frequency_mask3>(len_dataloader*5*5/10), 1, 0)

                        frequency_mask_path1 = frequency_mask_flod_path + 'frequency_mask1'
                        np.save(frequency_mask_path1, last_tensor_mask1.detach().cpu().numpy())

                        frequency_mask_path2 = frequency_mask_flod_path + 'frequency_mask2'
                        np.save(frequency_mask_path2, last_tensor_mask2.detach().cpu().numpy())

                        frequency_mask_path3 = frequency_mask_flod_path + 'frequency_mask3'
                        np.save(frequency_mask_path3, last_tensor_mask3.detach().cpu().numpy())


        else:
            frequency_mask1 = np.load('./save_masks/' + run_name + '/' + 'frequency_mask1.npy')
            frequency_mask2 = np.load('./save_masks/' + run_name + '/' + 'frequency_mask2.npy')
            frequency_mask3 = np.load('./save_masks/' + run_name + '/' + 'frequency_mask3.npy')
            last_tensor_mask1 = torch.from_numpy(frequency_mask1).cuda()
            last_tensor_mask2 = torch.from_numpy(frequency_mask2).cuda()
            last_tensor_mask3 = torch.from_numpy(frequency_mask3).cuda()


        k_space_r1 = torch.multiply(last_tensor[:, 0:1, :, :], last_tensor_mask1.unsqueeze(0))
        k_space_r2 = torch.multiply(last_tensor[:, 1:2, :, :], last_tensor_mask2.unsqueeze(0))
        k_space_r3 = torch.multiply(last_tensor[:, 2:3, :, :], last_tensor_mask3.unsqueeze(0))

        k_space_i1 = torch.multiply(last_tensor[:, 3:4, :, :], last_tensor_mask1.unsqueeze(0))
        k_space_i2 = torch.multiply(last_tensor[:, 4:5, :, :], last_tensor_mask2.unsqueeze(0))
        k_space_i3 = torch.multiply(last_tensor[:, 5:6, :, :], last_tensor_mask3.unsqueeze(0))

        x_real = torch.cat([k_space_r1, k_space_r2, k_space_r3], dim=1)
        x_imag = torch.cat([k_space_i1, k_space_i2, k_space_i3], dim=1)

        fft_src_ = torch.complex(x_real, x_imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1))
        src_in_trg = torch.real(src_in_trg)
        x_ = src_in_trg.float()


        b1,b2,b3,b4,b5,b6 = self.encoder_segment(x_)
        output_segment = self.decoder_segment(b1,b2,b3,b4,b5,b6)
        if self.out_features:
            return output_segment, b2, b3, b4, b5, b6
        else:
            return output_segment, last_tensor_mask1, last_tensor_mask2, last_tensor_mask3, x_


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))

        self.EFCm1 = EFCm(base_width, base_width, 3,
                        0.5, 0.5, stride=1, padding=1)

        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))

        self.EFCm2 = EFCm(base_width * 2, base_width * 2, 3,
                        0.5, 0.5, stride=1, padding=1)

        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))

    def forward(self, x):

        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        mp1 = self.EFCm1(mp1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        mp2 = self.EFCm2(mp2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()

        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                  nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(base_width * 8),
                                  nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 8), base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 4), base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b, b5), dim=1)
        db_b = self.db_b(cat_b)
        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.db1(cat1)
        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.db2(cat2)
        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.db3(cat3)
        up4 = self.up4(db3)
        cat4 = torch.cat((up4, b1), dim=1)
        db4 = self.db4(cat4)
        out = self.fin_out(db4)

        return out



import torch
import math
import functools
import torch.nn.functional as F
from models.NEDB_IN import Dense_Block_IN,Dense_Block_IN
from torch import nn
from torch.nn import Parameter
import numpy as np
# from models.base_network import BaseNetwork
# from .blocks import ConvBlock, DeconvBlock, MeanShift
class AttBlock(nn.Module):
	def __init__(self):
		super(AttBlock, self).__init__()
		self.softmax = nn.Softmax()

	def forward(self, fq, fk):
		"""Forward function (with skip connections)"""
		w = self.softmax(fq * fk)
		out = w * (fq + fk)
		return out

class SAFA(nn.Module):
	def __init__(self):
		super(SAFA, self).__init__()
		self.conv_d4_1 = nn.Conv2d(24, 24, kernel_size=7, stride=4, padding=2)
		self.conv_d2_1 = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1)
		self.conv_d2_2 = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1)
		self.att0 = AttBlock()
		self.att1 = AttBlock()
		self.att2 = AttBlock()
		self.att3 = AttBlock()
		self.att4 = AttBlock()
		self.att5 = AttBlock()
		self.att6 = AttBlock()
		self.att7 = AttBlock()

	def forward(self, F, Fq):
		"""Forward function (with skip connections)"""
		Q = self.conv_d2_1(self.conv_d4_1(F))
		K = self.conv_d2_2(Fq)
		Q_N = torch.chunk(Q, 8, dim=1)
		K_N = torch.chunk(K, 8, dim=1)
		F_list = []
		F_list.append(self.att0(Q_N[0], K_N[0]))
		F_list.append(self.att1(Q_N[1], K_N[1]))
		F_list.append(self.att2(Q_N[2], K_N[2]))
		F_list.append(self.att3(Q_N[3], K_N[3]))
		F_list.append(self.att4(Q_N[4], K_N[4]))
		F_list.append(self.att5(Q_N[5], K_N[5]))
		F_list.append(self.att6(Q_N[6], K_N[6]))
		F_list.append(self.att7(Q_N[7], K_N[7]))
		out = torch.cat([F_list[0],F_list[1],F_list[2],F_list[3],F_list[4],F_list[5],F_list[6],F_list[7]], dim=1)
		return out

class FEN(nn.Module):
	def __init__(self):
		super(FEN,self).__init__()
		self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
		self.conv1 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(True)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(True)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(True)
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(True)
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.ReLU(True)
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(64, 24, 3, 1, 1),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.conv0(x)
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2)
		x4 = self.conv4(x3)
		x5 = self.conv5(torch.cat([x4,x2], dim=1))
		x6 = self.conv6(torch.cat([x5,x1], dim=1))
		return x6

class FeaEnhancer(nn.Module):
	def __init__(self):
		super(FeaEnhancer, self).__init__()
		self.conv_d4 = nn.Conv2d(3, 3, kernel_size=7, stride=4, padding=2)
		self.conv_d2 = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
		self.FEN = FEN()
		self.SAFA = SAFA()
		self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
		self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
		self.conv = nn.Conv2d(48, 3, kernel_size=3, stride=1, padding=1)

	def forward(self, I):
		Iq = self.conv_d4(I)
		Io = self.conv_d2(Iq)
		F = self.FEN(I)
		Fq = self.FEN(Iq)
		Fo = self.FEN(Io)
		Fh = self.SAFA(F, Fq)
		Fh_up = self.up1(Fh)
		Fo_up = self.up2(Fo)
		res = self.conv(torch.cat([Fh_up,Fo_up], dim=1))
		out = I + res
		return res, out

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, input_dim, output_dim):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
			nn.Conv2d(input_dim, output_dim, 3, 1, 1),
			nn.InstanceNorm2d(output_dim),
			nn.ReLU(True),
			nn.Conv2d(output_dim, output_dim, 3, 1, 1),
			nn.InstanceNorm2d(output_dim),
			nn.ReLU(True),
		)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Trans_Up(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Up, self).__init__()
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=2, mode='bilinear')
		out = self.relu(self.IN1(self.conv0(x)))
		return out

class Trans_Down(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Down, self).__init__()
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.IN1(self.conv0(x)))
		return out


import torch
import torch.nn as nn


class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)


	def forward(self, x):
		out = torch.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += x
		out = torch.relu(out)
		return out

class FE(nn.Module):
	def __init__(self):
		super(FE, self).__init__()

		self.conv11 = nn.Sequential(
			ResBlock(64, 64),  # 256 -> 128
			ResBlock(64, 64),
			ResBlock(64, 64),
			ResBlock(64, 64),
		)
		self.conv12 = nn.Sequential(
			ResBlock(128, 128),  # 256 -> 128
			ResBlock(128, 128),
			ResBlock(128, 128),
			ResBlock(128, 128),
		)
		self.conv13 = nn.Sequential(
			ResBlock(256, 256),  # 256 -> 128
			ResBlock(256, 256),
			ResBlock(256, 256),
			ResBlock(256, 256),
		)
		#
		# 几个 conv, 中间 channel, 输入 channel

	#

	def forward(self, x):  # 1, 3, 256, 256   300
		#
		[d1, d2, d3] = x
		#######################################################
		d1 = self.conv11(d1)   # 64,256,256
		d2 = self.conv12(d2)   # 64, 128, 128
		d3 = self.conv13(d3)    # 64, 64, 64

		return [d1, d2, d3]

class SAD(nn.Module):
	def __init__(self):
		super(SAD, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(256, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		# 几个 conv, 中间 channel, 输入 channel
		self.Dense_Up_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 256 -> 128
		self.Dense_Up_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 128 -> 64
		self.Dense_Up_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 64 -> 32
		self.Dense_Up_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 32 -> 16
		#
		self.Latent = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 16 -> 8
		#
		self.Dense_Down_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 16
		self.Dense_Down_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 32
		self.Dense_Down_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 64
		self.Dense_Down_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 128
		# 下采样
		self.trans_down_1 = Trans_Down(64, 64)
		self.trans_down_2 = Trans_Down(64, 64)
		self.trans_down_3 = Trans_Down(64, 64)
		self.trans_down_4 = Trans_Down(64, 64)
		# 上采样
		self.trans_up_4 = Trans_Up(64, 64)
		# self.trans_up_3 = Trans_Up_1(64, 64)
		self.trans_up_3 = Trans_Up(64, 64)
		self.trans_up_2 = Trans_Up(64, 64)
		self.trans_up_1 = Trans_Up(64, 64)
		# 融合
		self.up_4_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_3_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_2_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_1_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
		)
		self.fusion2 = nn.Sequential(
			nn.Conv2d(64, 3, 3, 1, 1),
			nn.Tanh(),
		)

	#

	def forward(self, x):  # 1, 3, 256, 256   300
		#
		[d1, d2, d3] = x
		#######################################################
		d1 = self.conv1(d1)   # 64,256,256
		d2 = self.conv2(d2)   # 64, 128, 128
		d3 = self.conv3(d3)    # 64, 64, 64
		down_11 = self.Dense_Down_1(d1)  # 1, 64, 256, 256

		down_21 = self.Dense_Down_2(d2)  # 1, 64, 128, 128

		down_31 = self.Dense_Down_3(d3)  # 1, 64, 64, 64
		down_3 = self.trans_down_3(down_31)  # 1, 64, 32, 32

		down_41 = self.Dense_Down_4(down_3)  # 1, 64, 32, 32
		down_4 = self.trans_down_4(down_41)  # 1, 64, 16, 16

		#######################################################
		Latent = self.Latent(down_4)  # 1, 64, 16, 16               19
		#######################################################

		up_4 = self.trans_up_4(Latent)  # 1, 64, 32, 32             38
		up_4 = torch.cat([down_41, up_4], dim=1)  # 1, 128, 32, 32  38
		up_4 = self.up_4_fusion(up_4)  # 1, 64, 32, 32           38
		up_4 = self.Dense_Up_4(up_4)  # 1, 64, 32, 32          38

		up_3 = self.trans_up_3(up_4)  # 1, 64, 64, 64               76
		up_3 = torch.cat([down_31, up_3], dim=1)  # 1, 128, 64, 64
		up_3 = self.up_3_fusion(up_3)  # 1, 64, 64, 64
		up_3 = self.Dense_Up_3(up_3)  # 1, 64, 64, 64

		up_2 = self.trans_up_2(up_3)  # 1, 64, 128,128
		up_2 = torch.cat([up_2, down_21], dim=1)  # 1, 128, 128,128
		up_2 = self.up_2_fusion(up_2)  # 1, 64, 128,128
		up_2 = self.Dense_Up_2(up_2)  # 1, 64, 128,128

		up_1 = self.trans_up_1(up_2)  # 1, 64, 256, 256
		up_1 = torch.cat([up_1, down_11], dim=1)  # 1, 128, 256, 256
		up_1 = self.up_1_fusion(up_1)  # 1, 64, 256, 256
		up_1 = self.Dense_Up_1(up_1)  # 1, 64, 256, 256
		#
		feature = self.fusion(up_1)  # 1, 64, 256, 256
		#
		# feature = feature + feature_neg  # 1, 64, 256, 256
		#
		outputs = self.fusion2(feature)
		return outputs

class NLEDN_IN_32_16_32(nn.Module):
	def __init__(self, input_nc, output_n):
		super(NLEDN_IN_32_16_32, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(input_nc, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		# 几个 conv, 中间 channel, 输入 channel
		self.Dense_Up_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 256 -> 128
		self.Dense_Up_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 128 -> 64
		self.Dense_Up_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 64 -> 32
		self.Dense_Up_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 32 -> 16
		#
		self.Latent = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 16 -> 8
		#
		self.Dense_Down_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 16
		self.Dense_Down_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 32
		self.Dense_Down_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 64
		self.Dense_Down_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 128
		# 下采样
		self.trans_down_1 = Trans_Down(64, 64)
		self.trans_down_2 = Trans_Down(64, 64)
		self.trans_down_3 = Trans_Down(64, 64)
		self.trans_down_4 = Trans_Down(64, 64)
		# 上采样
		self.trans_up_4 = Trans_Up(64, 64)
		# self.trans_up_3 = Trans_Up_1(64, 64)
		self.trans_up_3 = Trans_Up(64, 64)
		self.trans_up_2 = Trans_Up(64, 64)
		self.trans_up_1 = Trans_Up(64, 64)
		# 融合
		self.up_4_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_3_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_2_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_1_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
		)
		self.fusion2 = nn.Sequential(
			nn.Conv2d(64, output_n, 3, 1, 1),
			nn.Tanh(),
		)
		#

	def forward(self, x):   # 1, 3, 256, 256   300
		#
		feature_neg = self.conv1(x)   # 1, 64, 256, 256    300
		feature_0 = self.conv2(feature_neg)   # # 1, 64, 256, 256   300
		#######################################################
		down_11 = self.Dense_Down_1(feature_0)  # 1, 64, 256, 256   300
		down_1 = self.trans_down_1(down_11)   # 1, 64, 128, 128     150

		down_21 = self.Dense_Down_2(down_1)  # 1, 64, 128, 128      150
		down_2 = self.trans_down_2(down_21)    # 1, 64, 64, 64      75

		down_31 = self.Dense_Down_3(down_2)  # 1, 64, 64, 64        75
		down_3 = self.trans_down_3(down_31)    # 1, 64, 32, 32      38

		down_41 = self.Dense_Down_4(down_3)  # 1, 64, 32, 32        38
		down_4 = self.trans_down_4(down_41)    # 1, 64, 16, 16      19

		#######################################################
		Latent = self.Latent(down_4)  # 1, 64, 16, 16               19
		#######################################################

		up_4 = self.trans_up_4(Latent)  # 1, 64, 32, 32             38
		up_4 = torch.cat([down_41, up_4], dim=1)  # 1, 128, 32, 32  38
		up_4 = self.up_4_fusion(up_4)     # 1, 64, 32, 32           38
		up_4 = self.Dense_Up_4(up_4)       # 1, 64, 32, 32          38

		up_3 = self.trans_up_3(up_4)  # 1, 64, 64, 64               76
		up_3 = torch.cat([down_31, up_3], dim=1)  # 1, 128, 64, 64
		up_3 = self.up_3_fusion(up_3)     # 1, 64, 64, 64
		up_3 = self.Dense_Up_3(up_3)       # 1, 64, 64, 64

		up_2 = self.trans_up_2(up_3)  # 1, 64, 128,128
		up_2 = torch.cat([up_2, down_21], dim=1)  # 1, 128, 128,128
		up_2 = self.up_2_fusion(up_2)     # 1, 64, 128,128
		up_2 = self.Dense_Up_2(up_2)       # 1, 64, 128,128

		up_1 = self.trans_up_1(up_2)  # 1, 64, 256, 256
		up_1 = torch.cat([up_1, down_11], dim=1)  # 1, 128, 256, 256
		up_1 = self.up_1_fusion(up_1)     # 1, 64, 256, 256
		up_1 = self.Dense_Up_1(up_1)       # 1, 64, 256, 256
		#
		feature = self.fusion(up_1)  # 1, 64, 256, 256
		#
		feature = feature + feature_neg  # 1, 64, 256, 256
		#
		outputs = self.fusion2(feature)
		return outputs
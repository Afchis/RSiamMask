import torch
import torch.nn as nn
import torch.nn.functional as F

from args import *
from resnet_afchi import *


'''
Siam parts
'''
def Correlation_func(t_f, s_f): # s_f-->search_feat, t_f-->target_feat
	t_f = t_f.reshape(-1, 1, t_f.size(2), t_f.size(3))
	out = s_f.reshape(1, -1, s_f.size(2), s_f.size(3)) # 1, b*ch, 32, 32
	out = F.conv2d(out, t_f, groups=t_f.size(0))
	out = out.reshape(-1, s_f.size(1), out.size(2), out.size(3))
	return out


class Backbone(nn.Module):
	def __init__(self):
		super(Backbone, self).__init__()
		self.model = resnet50(pretrained=True)
		self.adjust = nn.Conv2d(1024, 256, kernel_size=1) 

	def forward(self, x):
		search_cat = self.model(x)
		out = search_cat[4]
		if out.size(3) < 20:
			out = out[:, :, 4:-4, 4:-4]
		out = self.adjust(out)
		return search_cat, out


class ScoreBranch(nn.Module):
	def __init__(self):
		super(ScoreBranch, self).__init__()
		self.conv_target = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			)
		self.conv_searchs = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			)
		self.branch = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 1, kernel_size=1),
			)

	def forward(self, target_feat, searchs_feat):
		t_f = self.conv_target(target_feat)
		s_f = self.conv_searchs(searchs_feat)
		out = Correlation_func(t_f, s_f)
		out = self.branch(out)
		return out


class MaskBranch(nn.Module):
	def __init__(self):
		super(MaskBranch, self).__init__()
		self.conv_target = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			)
		self.conv_searchs = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			)
		self.branch = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 63*63, kernel_size=1),
			)

	def forward(self, target_feat, searchs_feat):
		t_f = self.conv_target(target_feat)
		s_f = self.conv_searchs(searchs_feat)
		out = Correlation_func(t_f, s_f)
		out = self.branch(out)
		return out


if __name__ == '__main__':
	model = Backbone()
	tensor = torch.rand([BATCH_SIZE, 3, 128, 128])
	search_cat, out = model(tensor)
	print(out.shape)
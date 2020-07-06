import torch
import torch.nn as nn

from args import *
from resnet import *


'''
Siam parts
'''
class Backbone(nn.Module):
	def __init__(self):
		super(Backbone, self).__init__()
		self.model = resnet50(pretrained=True)
		self.adjust = nn.Conv2d(1024, 256, kernel_size=1) 

	def forward(self, x):
		search_cat = self.model(x)
		out = search_cat[4]
		if out.size(3) < 20:
			# print(out.shape)
			out = out[:, :, 4:-4, 4:-4]
			# print(out.shape)
		out = self.adjust(out)
		return search_cat, out


class ScoreBranch(nn.Module):
	def __init__(self):
		super(ScoreBranch, self).__init__()
		self.branch = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 1, kernel_size=1),
			nn.Sigmoid()
			)

	def forward(self, x):
		score = self.branch(x)
		pos_list = torch.tensor([], dtype=int).to(device)
		for i in range(score.size(0)):
			max_value = score[i][0].max()
			pos = (score[i] == max_value).nonzero()[0][1:].unsqueeze(0)
			pos_list = torch.cat([pos_list, pos], dim=0)
		return score, pos_list		


# class MaskBranch(nn.Module):
# 	def __init__(self):
# 		super(MaskBranch, self).__init__()
# 		self.deconv = nn.ConvTranspose2d(256, 32, 16, 16)
# 		self.branch = nn.Conv2d(32, 32, kernel_size=1)

# 	def forward(self, masks_feat):
		
# 		out = self.deconv(masks_feat)
# 		# out = out.reshape(TIMESTEPS*BATCH_SIZE, 32, 16, 16)
# 		return out

class MaskBranch(nn.Module):
	def __init__(self):
		super(MaskBranch, self).__init__()
		self.branch = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1),
			# nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 64*64, kernel_size=1),
			)

	def forward(self, masks_feat):
		# print(masks_feat.shape)
		out = self.branch(masks_feat)
		# print(out.shape)
		return out

# class MaskBranch3(nn.Module):
# 	def __init__(self):
# 		super(MaskBranch3, self).__init__() 
# 		self.deconv = nn.ConvTranspose2d(256, 2, 64, 64)

# 	def forward(self, masks_feat):
# 		out = self.deconv(masks_feat)
# 		return out


if __name__ == '__main__':
	model = Backbone()
	tensor = torch.rand([BATCH_SIZE, 3, 128, 128])
	search_cat, out = model(tensor)
	print(out.shape)
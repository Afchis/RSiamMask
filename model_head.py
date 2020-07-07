import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from args import *
from model_parts import *


'''
Model head
'''
class ModelDisigner(nn.Module):
	def __init__(self):
		super(ModelDisigner, self).__init__()
		self.backbone = Backbone()
		self.score_branch = ScoreBranch()
		self.mask_branch = MaskBranch()
		self.up = nn.Upsample(scale_factor=2, mode='nearest')
		self.sig = nn.Sigmoid()

		self.training = True

	def forward(self, target, searchs, score_label, mask_label):
		# mask_label = mask_label[:, :, 64:-64, 64:-64]
		_,  target_feat = self.backbone(target)
		search_cats, searchs_feat = self.backbone(searchs)
		##### Score Branch #####
		score = self.score_branch(target_feat, searchs_feat)
		score = self.sig(score)
		##### Mask Branch #####
		masks = self.mask_branch(target_feat, searchs_feat)
		# masks = self.sig(masks)
		# masks = self.up(masks)
		##### Losses #####
		score_loss = self.score_loss_bce(score, score_label)
		mask_loss, afchi_label, afchi_mask = self.select_mask_logistic_loss(masks, mask_label, score_label)
		return score, afchi_mask, score_loss, mask_loss, afchi_label

	def get_cls_loss(self, pred, label, select):
		if select.nelement() == 0: return pred.sum()*0.
		pred = torch.index_select(pred, 0, select)
		label = torch.index_select(label, 0, select)
		return F.cross_entropy(pred, label)

	def score_loss(self, score, score_label):
		# score = F.log_softmax(score, dim=1)
		score_label = score_label.view(-1)
		score = score.view(-1, 2)
		pos = Variable(score_label.data.eq(1).nonzero().squeeze()).cuda()
		neg = Variable(score_label.data.eq(0).nonzero().squeeze()).cuda()

		loss_pos = self.get_cls_loss(score, score_label.long(), pos)
		loss_neg = self.get_cls_loss(score, score_label.long(), neg)
		loss = 0.5*loss_pos + 0.5*loss_neg
		return loss

	def select_mask_logistic_loss(self, p_m, mask, weight, o_sz=63, g_sz=127):
		weight = weight.view(-1)
		pos = Variable(weight.data.eq(1).nonzero().squeeze())
		if pos.nelement() == 0: return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

		p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
		p_m = torch.index_select(p_m, 0, pos)

		p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
		p_m = p_m.view(-1, g_sz * g_sz)

		mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
		mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)

		mask_uf = torch.index_select(mask_uf, 0, pos)
		afchi_label = mask_uf
		afchi_mask = (p_m >= 0).float()
		loss = F.soft_margin_loss(p_m, mask_uf)
		return loss, afchi_label, afchi_mask

	def score_loss_bce(self, score, score_label):
		return F.binary_cross_entropy(score, score_label)

	def mask_loss_bce(self, masks, mask_label):
		return F.binary_cross_entropy(score, score_label)








if __name__ == '__main__':
	model = ModelDisigner()
	model = model.to(device)
	target = torch.rand([BATCH_SIZE, 3, 128, 128]).to(device)
	searchs = torch.rand([BATCH_SIZE, 3, 256, 256]).to(device)
	score, mask = model(target, searchs)
	print('score.shape: ', score.shape)
	print('mask.shape: ', mask.shape)

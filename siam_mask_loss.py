import torch
import torch.nn.functional as F
from torch.autograd import Variable


def get_cls_loss(pred, label, select):
    if select.nelement() == 0: return pred.sum()*0.
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)

    return F.nll_loss(pred, label)

def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
    neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
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

    loss = F.soft_margin_loss(p_m, mask_uf)
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    return loss, iou_m, iou_5, iou_7#, afchi
import torch.nn as nn
from component.loss import OhemCrossEntropy2d
import torch
from torch.nn import functional as F
from component.lovasz_losses import lovasz_softmax


class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor(
            [0.5, 0.9, 0.9, 1.1, 1.0,
             0.9, 1.3, 0.9, 1.0, 0.9,
             1.4, 2, 1.6, 0.9, 0.9,
             1, 1.2, 1.2, 1.3, 1.3])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index,weight=weight)

    def forward(self, preds, target):
        scale_pred = F.interpolate(preds, scale_factor=4., mode='bilinear', align_corners=False)
        loss = self.criterion(scale_pred, target)
        return loss


class CriterionAll_lovasz(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll_lovasz, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=False)
        loss = lovasz_softmax(F.softmax(scale_pred, dim=1), target)
        return loss


class Criterion_lovasz_ce(nn.Module):
    def __init__(self, ignore_index=255):
        super(Criterion_lovasz_ce, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor(
            [0.8, 0.9, 0.9, 1.1, 1.8,
             0.9, 1.3, 0.9, 0.9, 0.9,
             2, 2, 1.4, 0.9, 0.9,
             0.9, 1.2, 1.2, 1.3, 1.3])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=False)
        loss = 0.3 * self.criterion(scale_pred, target) + 0.7 * lovasz_softmax(F.softmax(scale_pred, dim=1), target)
        return loss


class CriterionOhemDSN(nn.Module):
    '''
    DSN + OHEM : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, dsn_weight=0.4, use_weight=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=use_weight)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=False)
        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=False)
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CriterionOhemDSN_single(nn.Module):
    '''
    DSN + OHEM : we find that use hard-mining for both supervision harms the performance.
                Thus we choose the original loss for the shallow supervision
                and the hard-mining loss for the deeper supervision
    '''

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, dsn_weight=0.4):
        super(CriterionOhemDSN_single, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor(
            [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
             0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.criterion_ohem = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=True)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=False)
        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=False)
        loss2 = self.criterion_ohem(scale_pred, target)
        return self.dsn_weight * loss1 + loss2

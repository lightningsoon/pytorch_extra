'''
多尺度模块，把模型包裹进来就可以，输出和以前一样
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSC(nn.Module):
    """Multi-scale inputs"""

    def __init__(self, model, ori_size,pyramids=(336, )):
        super(MSC, self).__init__()
        self.model = model
        self.pyramids = pyramids
        self.ori_size=ori_size# int or tuple(int,int)

    def forward(self, x):
        # if MultiScale is false, only use the original input
        if len(self.pyramids)==1:
            return self.model(x) #输出和正常并行模型中的单卡一样的 [[粗轮廓,细轮廓],[边缘]] 细轮廓的shape = [BS,C,H,W]
        # Original
        # 多尺度变换
        #   预测 并恢复尺度
        # 取概率最大的那点
        y_pyramid = []
        for p in self.pyramids:
            x_inter=F.interpolate(x, size=(p,p), mode='bilinear', align_corners=True)
            y=self.model(x_inter)[0][1]#多输出，对于评估和测试只关心这个数据
            y=F.interpolate(y,size=self.ori_size,mode='bilinear', align_corners=True)
            y_pyramid.append(y)
        if len(self.pyramids)>=1:
            y = torch.mean(torch.stack(tuple(y_pyramid)), dim=(0,))#[0]# 第一个是值，第二个位置
        return [[None,y],[None]]

# 翻转预测，多尺度
# 集成预测
import torch
import torch.nn
import numpy as np

def filp_inf(model: torch.nn.Module, img: torch.Tensor)->list:

    pred0 = model(img.cuda())
    img=img.cpu().numpy()
    pred1 = model(torch.from_numpy(np.flip(img,axis=3).copy()).cuda())
    pred=[]
    if type(pred0) != list:
        pred0=[pred0]
        pred1=[pred1]
    for p0,p1 in zip(pred0,pred1):
        p1 = p1.detach().cpu().numpy()
        p1 = p1[:, :, :, ::-1].copy()# torch dont support reverse of numpy
        p1_copy = p1[:, 14:].copy()
        p1[:, 14], p1[:,15], \
        p1[:,16], p1[:,17], \
        p1[:,18], p1[:,19] = p1_copy[:,1], p1_copy[:,0], \
                                 p1_copy[:,3], p1_copy[:,2], \
                                 p1_copy[:,5], p1_copy[:,4]

        p0 = p0.detach().cpu().numpy()
        p=(p0 + p1) / 2 # 同设备要求
        p=torch.from_numpy(p)
        pred.append(p)
    return pred

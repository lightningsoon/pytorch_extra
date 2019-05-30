from torch.utils import data
from component.data_process import Process
import os
import cv2
import numpy as np
from functools import partial
# from nori2 import Fetcher
import pickle
# from meghair.utils.imgproc import imdecode
swapAxes4tensor = partial(np.transpose, axes=(2, 0, 1))

class BaseData(data.Dataset, Process):
    def __init__(self, root, status, size, scale=0., rotation=0):
        data.Dataset.__init__(self)
        Process.__init__(self,size, scale, rotation)
        # Fetcher.__init__(self)
        # nori2.Fetcher.__init__(self)
        # 根据索引找文件
        self.root = root
        self.status = status
        list_path = os.path.join(root, self.status + '_id.txt')
        self.img_list = [i_id.strip() for i_id in open(list_path)][:]
        self.x_path = os.path.join(root, self.status + '_images')
        self.y_path = os.path.join(self.root, status + '_segmentations')
        # self.nori_dict=pickle.load(open(os.path.join(root ,status + '_data.pkl'), 'rb'))

    def __len__(self):
        return len(self.img_list)

    def read_x(self, index):
        # x_pth=self.nori_dict[self.img_list[index]]['image_nori_id']
        # img=imdecode(self.get(x_pth)).astype(np.uint8)
        filename = os.path.join(self.x_path, self.img_list[index] + '.jpg')
        img = cv2.imread(filename)
        return img, self.img_list[index]
    def read_y(self,index):
        # y_pth = self.nori_dict[self.img_list[index]]['label_nori_id']
        # y = imdecode(self.get(y_pth))[:, :, 0]
        yname = os.path.join(self.y_path, self.img_list[index] + '.png')
        y=cv2.imread(yname, cv2.IMREAD_GRAYSCALE)
        return y
        # return label
    def deal_item(self,item):
        trans_inv=1
        x, name = self.read_x(item)
        (h, w) = x.shape[:2]
        x = self.scale(x)
        # x, trans_inv = self.affine(x)
        x = self.normalize(x)
        x = np.transpose(x, (2, 0, 1))
        return x, name, (w, h), trans_inv
class TestData(BaseData):
    # 测试数据
    # 只要输入x和process
    def __init__(self, root, size, scale=0, rotation=0):
        super(TestData, self).__init__(root, 'test', size, scale, rotation)
        pass

    def __getitem__(self, item):
        x, name, (w, h), trans_inv=self.deal_item(item)
        return np.float32(x), name, (w, h), trans_inv


class valData(BaseData):
    # 验证数据
    # 需要对x，y都做仿射变换
    def __init__(self, root, size, scale=0, rotation=0, *args,**kwargs):
        status = kwargs.get('status', 'val')
        super(valData, self).__init__(root, status, size, scale, rotation)

        self.y_path = os.path.join(self.root, status + '_segmentations')

    def __getitem__(self, index):
        x,name,(w,h),trans_inv=self.deal_item(index)
        y_name = os.path.join(self.y_path, self.img_list[index] + '.png')
        return np.float32(x), name, (w, h), trans_inv, y_name


class trainData(BaseData):
    # 还有翻转、亮度、色彩、饱和度增强可以调
    #
    def __init__(self, root, size, scale, rotation, *args,**kwargs):
        status = kwargs.get('status', 'train')
        super(trainData, self).__init__(root, status, size, scale, rotation)
        self.y_path = os.path.join(self.root, status + '_segmentations')


    def __getitem__(self, index):
        x, name = self.read_x(index)
        y = self.read_y(index)
        x, y = self.flip(x, y=y)

        # x, trans_inv, y = self.affine(x, y)
        x, y = self.scale(x, y)
        # edge = self.generate_edge(y)
        x = self.normalize(x)
        x = swapAxes4tensor(x)
        return np.float32(x), np.int64(y), name

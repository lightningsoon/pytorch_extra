import cv2
import numpy as np
import random

class Process():
    def __init__(self, size, scale_factor=0, rotation=0):
        # 数据集处理的函数，需要用到请实例化并传给数据生成器
        # 只处理单图，或单对图
        # =0 表示取消
        self.transform_function = []
        self.__mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.__std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.__size = size
        self.aspect_ratio = self.__size[1] * 1.0 / self.__size[0]  # 宽高比
        self.__scale_factor = scale_factor
        self.__rotation = rotation
        self.__flip_probablity=0.5
        self.__flip_pairs = [[14, 16, 18],[15, 17, 19]]# 左右
        cv2.setNumThreads(0)#刚加的在测试
        pass

    def normalize(self, img):
        img = ((img / 255.) - self.__mean) / self.__std
        return img

    def flip(self, x, y=None):
        '''

        :param img:
        :param flipCode: 0 x轴，>0 y轴，<0 xy轴
        :return:
        '''
        if random.random()<self.__flip_probablity:
            x = cv2.flip(x, flipCode=1)
            if np.any(y!=None):
                y = cv2.flip(y, flipCode=1)
                for i in range(0, 3):
                    right_pos = np.where(y == self.__flip_pairs[1][i])
                    left_pos = np.where(y == self.__flip_pairs[0][i])
                    y[right_pos] = self.__flip_pairs[0][i]
                    y[left_pos] = self.__flip_pairs[1][i]
                return x,y
            return x
        else:
            if np.any(y!=None):
                return x,y
            else:
                return x

    def affine(self, img, y=None):
        h, w = img.shape[:2]

        # 仿射变换
        def box2cs(h, w):
            x, y, w, h = [0, 0, w - 1, h - 1]
            center = np.zeros((2), dtype=np.float32)
            center[0] = x + w * 0.5
            center[1] = y + h * 0.5
            # 用增加的方式，把短的边拉长
            if w > self.aspect_ratio * h:
                h = w * 1.0 / self.aspect_ratio
            elif w < self.aspect_ratio * h:
                w = h * self.aspect_ratio
            scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
            return center, scale

        def get_factor():
            rotation = 0
            center, scale = box2cs(h, w)
            if self.__rotation != 0:
                rotation = np.clip(np.random.randn() * self.__rotation, -self.__rotation * 2,
                                          self.__rotation * 2) \
                    if random.random() <= 0.6 else 0
            if self.__scale_factor != 0:
                scale = scale * np.clip(np.random.randn() * self.__scale_factor + 1, 1 - self.__scale_factor,
                                                      1 + self.__scale_factor)
            return rotation, center, scale

        rf, c, sf = get_factor()
        trans, trans_inv = get_affine_transform(c, sf, rf, self.__size)
        img = cv2.warpAffine(
            img,
            trans,
            (int(self.__size[1]), int(self.__size[0])),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        if np.any(y!=None):
            # 有没有y
            y = cv2.warpAffine(y,
                               trans,
                               (int(self.__size[1]), int(self.__size[0])),
                               flags=cv2.INTER_NEAREST,# 注意
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255,))

            return img, trans_inv, y
        else:
            return img, trans_inv

    def generate_edge(self,label, edge_width=3):
        h, w = label.shape
        edge = np.zeros(label.shape)

        # right
        edge_right = edge[1:h, :]
        edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
                   & (label[:h - 1, :] != 255)] = 1

        # up
        edge_up = edge[:, :w - 1]
        edge_up[(label[:, :w - 1] != label[:, 1:w])
                & (label[:, :w - 1] != 255)
                & (label[:, 1:w] != 255)] = 1

        # upright
        edge_upright = edge[:h - 1, :w - 1]
        edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                     & (label[:h - 1, :w - 1] != 255)
                     & (label[1:h, 1:w] != 255)] = 1

        # bottomright
        edge_bottomright = edge[:h - 1, 1:w]
        edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                         & (label[:h - 1, 1:w] != 255)
                         & (label[1:h, :w - 1] != 255)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
        edge = cv2.dilate(edge, kernel)
        return edge
    def rotate(self):
        # 旋转
        return

    def scale(self,x,y=None):
        x=cv2.resize(x,self.__size,interpolation=cv2.INTER_LINEAR)
        if np.any(y!=None):
            y=cv2.resize(y,self.__size,interpolation=cv2.INTER_NEAREST)
            return x,y
        return x
    def label_center_add_weight(self,y:np.ndarray,num_class=20)->np.ndarray:
        weight=np.ones((num_class,*y.shape),dtype=np.float32)
        for i in range(1,num_class):
            idx=np.where(y==i)
            if len(idx[0])>0:
                idx=np.array(idx)# shape 2,x
                g=np.mean(idx,axis=1,keepdims=True)# h,w shape=2,1 重心
                l=np.linalg.norm(idx-g, axis=0)
                # TODO 用高斯算权重，然后加一，赋值
                pass

        return weight


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    def get_dir(src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, trans_inv


# def transform_parsing(pred, center, scale, width, height, input_size):
#     trans = get_affine_transform(center, scale, 0, input_size, inv=1)
#     target_pred = cv2.warpAffine(
#         pred,
#         trans,
#         (int(width), int(height)),  # (int(width), int(height)),
#         flags=cv2.INTER_NEAREST,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=(0))
#
#     return target_pred

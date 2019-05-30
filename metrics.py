import numpy as np


def cal_ConfusionMatrix(y_true, y_pred, classes):
    ignore_index = y_true != 255

    y_true = y_true[ignore_index]
    y_pred = y_pred[ignore_index]
    y_true = np.int32(y_true)
    y_pred = np.int32(y_pred)  # 转int32防止溢出
    index = (y_true * classes + y_pred)  # 将每一段classes长度的数组，作为true标签与pred的对应关系，
    # 比如值20=1（true）*20+0（pred）唯一成立
    label_count = np.bincount(index.ravel())
    confusion_matrix = np.zeros((classes, classes),dtype=np.int32)
    L=len(label_count)
    for i_label in range(classes):
        for i_y_pred in range(classes):
            cur_index = i_label * classes + i_y_pred
            if cur_index < L:
                confusion_matrix[i_label, i_y_pred] = label_count[cur_index]

    return confusion_matrix

def seg_metrics_from_confusionMatrix(cm: np.ndarray):
    '''

    :param cm:混淆矩阵
    :return: 像素精度，均像素精度，均IoU，加权IoU，类别IoU
    '''
    gt = cm.sum(1)
    pred = cm.sum(0)
    intersection = np.diag(cm)

    pixel_accuracy = (intersection.sum() / gt.sum())  # 像素精度，真正/全部标签
    mean_accuracy = (intersection / np.maximum(1.0, gt)).mean()
    IoU_array = (intersection / np.maximum(1.0, gt + pred - intersection))
    mean_IoU = IoU_array.mean()
    fwIoU = (gt / gt.sum()) * IoU_array
    fwIoU = np.sum(fwIoU)
    return pixel_accuracy, mean_accuracy, mean_IoU, fwIoU, IoU_array

"""
 @Time    : 2021/7/8 09:17
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : TCSVT2021_DCENet
 @File    : misc.py
 @Function: Useful Functions
 
"""
import numpy as np
import os
import skimage.io, skimage.color
import skimage.transform
import xlwt

################################################################
######################## Train & Test ##########################
################################################################
class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

################################################################
######################## Evaluation ############################
################################################################
def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(sheetname="sheet1", cell_overwrite_ok=True)

    j = 0
    for data in datas:
        for i in range(len(data)):
            sheet1.write(i, j, data[i])
        j = j + 1

    f.save(file_path)

def get_gt_mask(imgname, MASK_DIR):
    filestr = imgname[:-4]
    mask_folder = MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".png"
    mask = skimage.io.imread(mask_path)
    mask = np.where(mask == 255, 1, 0).astype(np.float32)

    return mask

def get_normalized_predict_mask(imgname, PREDICT_MASK_DIR):
    filestr = imgname[:-4]
    mask_folder = PREDICT_MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".png"
    if not os.path.exists(mask_path):
        print("{} has no predict mask!".format(imgname))
    mask = skimage.io.imread(mask_path).astype(np.float32)
    if np.max(mask) > 0:
        mask = (mask - np.min(mask))/(np.max(mask) - np.min(mask))
    mask = mask.astype(np.float32)
    mask = skimage.color.rgb2grey(mask)

    return mask

def get_binary_predict_mask(imgname, PREDICT_MASK_DIR):
    filestr = imgname[:-4]
    mask_folder = PREDICT_MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".png"
    if not os.path.exists(mask_path):
        print("{} has no predict mask!".format(imgname))
    mask = skimage.io.imread(mask_path).astype(np.float32)
    mask = skimage.color.rgb2grey(mask)
    mask = np.where(mask >= 127.5, 1, 0).astype(np.float32)

    return mask

################################################################
######################## SOD Evaluation ########################
################################################################
def cal_precision_recall_mae(prediction, gt):
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4

    prediction = prediction / 255.
    gt = gt / 255.

    mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction >= threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae

def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure

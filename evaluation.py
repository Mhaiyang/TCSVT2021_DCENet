"""
 @Time    : 2021/7/8 09:07
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : TCSVT2021_DCENet
 @File    : evaluation.py
 @Function: Evaluation
 
"""
import os
import numpy as np
from PIL import Image
from collections import OrderedDict
from misc import *
from config import ecssd_path, hkuis_path, hkuis_test_path, pascals_path, sod_path, dutste_path, dutomron_path

results_path = './results'

to_test = OrderedDict([
    ('SOD', sod_path),
    ('PASCAL-S', pascals_path),
    ('DUT-OMRON', dutomron_path),
    ('ECSSD', ecssd_path),
    ('HKU-IS', hkuis_path),
    ('HKU-IS-TEST', hkuis_test_path),
    ('DUTS-TE', dutste_path),
])

print(results_path)
for key in to_test:
    print("{:12} {}".format(key, to_test[key]))

results = OrderedDict()

for name, root in to_test.items():
    prediction_path = os.path.join(results_path, name)
    gt_path = os.path.join(root, 'mask')

    precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
    mae_record = AvgMeter()

    img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_path) if f.endswith('.png')]
    for idx, img_name in enumerate(img_list):
        print('evaluating for %s: %d / %d      %s' % (name, idx + 1, len(img_list), img_name + '.png'))

        prediction = np.array(Image.open(os.path.join(prediction_path, img_name + '.png')).convert('L'))
        gt = np.array(Image.open(os.path.join(gt_path, img_name + '.png')).convert('L'))

        precision, recall, mae = cal_precision_recall_mae(prediction, gt)
        for idx, data in enumerate(zip(precision, recall)):
            p, r = data
            precision_record[idx].update(p)
            recall_record[idx].update(r)

        mae_record.update(mae)

    fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                            [rrecord.avg for rrecord in recall_record])

    results[name] = OrderedDict([('F', "%.3f" % fmeasure), ('mae', "%.3f" % mae_record.avg)])

print(results_path)
for key in results:
    print("{:12} {}".format(key, results[key]))


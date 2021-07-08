"""
 @Time    : 2021/7/7 21:40
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : TCSVT2021_DCENet
 @File    : config.py
 @Function: Configurations
 
"""
import os

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

datasets_root = '/media/iccd/disk1/saliency_benchmark'

sod_training_root = os.path.join(datasets_root, 'DUTS-TR')
msra10k_path = os.path.join(datasets_root, 'MSRA10K')
ecssd_path = os.path.join(datasets_root, 'ECSSD')
hkuis_path = os.path.join(datasets_root, 'HKU-IS')
hkuis_test_path = os.path.join(datasets_root, 'HKU-IS-TEST')
pascals_path = os.path.join(datasets_root, 'PASCAL-S')
dutomron_path = os.path.join(datasets_root, 'DUT-OMRON')
sod_path = os.path.join(datasets_root, 'SOD')
dutste_path = os.path.join(datasets_root, 'DUTS-TE')

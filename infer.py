"""
 @Time    : 2021/7/8 10:02
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : TCSVT2021_DCENet
 @File    : infer.py
 @Function: Inference
 
"""
import time
import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean

from config import *
from misc import *
from DCENet import DCENet

torch.manual_seed(2021)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

results_path = './results'
check_mkdir(results_path)
exp_name = 'DCENet'
args = {
    'scale': 320,
    'save_results': True,
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
    ('SOD', sod_path),
    ('PASCAL-S', pascals_path),
    ('DUT-OMRON', dutomron_path),
    ('ECSSD', ecssd_path),
    ('HKU-IS', hkuis_path),
    ('HKU-IS-TEST', hkuis_test_path),
    ('DUTS-TE', dutste_path),
])

results = OrderedDict()

def main():
    net = DCENet(backbone_path).cuda(device_ids[0])

    print('Load pre-trained model for testing')
    net.load_state_dict(torch.load('DCENet.pth'))
    print('Load DCENet.pth succeed!')

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():

            image_path = os.path.join(root, 'image')

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, '%s' % (name)))

            time_list = []
            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('g')]
            for idx, img_name in enumerate(img_list):
                if name == 'HKU-IS':
                    img = Image.open(os.path.join(image_path, img_name + '.png')).convert('RGB')
                else:
                    img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])

                start_each = time.time()
                prediction = net(img_var)
                time_list.append(time.time() - start_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))

                if args['save_results']:
                    Image.fromarray(prediction).convert('RGB').save(os.path.join(results_path, exp_name, '%s' % (name), img_name + '.png'))

            print("{}'s average Time Is : {:.1f} ms".format(name, 1000 * mean(time_list)))

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

if __name__ == '__main__':
    main()
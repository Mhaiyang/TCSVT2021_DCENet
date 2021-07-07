# TCSVT2021_DCENet

## Exploring Dense Context for Salient Object Detection
[Haiyang Mei](https://mhaiyang.github.io/), Yuanyuan Liu, Ziqi Wei, Dongsheng Zhou, Xiaopeng Wei, Qiang Zhang, and Xin Yang

[[Paper](https://ieeexplore.ieee.org/document/9389751)] [[Project Page](https://mhaiyang.github.io/TCSVT2021_TCSVTNet/index.html)]

### Abstract
Contexts play an important role in salient object detection (SOD). High-level contexts describe the relations between different parts/objects and thus are helpful for discovering the specific locations of salient objects while low-level contexts could provide the fine detail information for delineating the boundary of the salient objects. However, the way of perceiving/leveraging rich contexts has not been fully investigated by existing SOD works. The common context extraction strategies (e.g., leveraging convolutions with large kernels or atrous convolutions with large dilation rates) do not consider the effectiveness and efficiency simultaneously and may cause sub-optimal solutions. In this paper, we devote to exploring an effective and efficient way to learn rich contexts for accurate SOD. Specifically, we first build a dense context exploration (DCE) module to capture dense multi-scale contexts and further leverage the learned contexts to enhance the features discriminability. Then, we embed multiple DCE modules in an encoder-decoder architecture to harvest dense contexts of different levels. Furthermore, we propose an attentive skip-connection to transmit useful features from the encoder part to the decoder part for better dense context exploration. Finally, extensive experiments demonstrate that the proposed method achieves more superior detection results on the six benchmark datasets than 18 state-of-the-art SOD methods.

### Citation
If you use this code, please cite:

```
@article{Mei_2021_TCSVT,
    author = {Mei, Haiyang and Liu, Yuanyuan and Wei, Ziqi and Zhou, Dongsheng and Wei, Xiaopeng and Zhang, Qiang and Yang, Xin},
    title = {Exploring Dense Context for Salient Object Detection},
    booktitle = {IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)},
    year = {2021}
}
```

### Requirements
* PyTorch == 1.0.0
* TorchVision == 0.2.1
* CUDA 10.0  cudnn 7.2

### Test
Download 'resnet50-19c8e357.pth' at [here](https://download.pytorch.org/models/resnet50-19c8e357.pth) and trained model 'DCENet.pth' at [here](https://mhaiyang.github.io/TCSVT2021_DCENet/index.html), then run `infer.py`.

### License
Please see `License.txt`

### Contact
E-Mail: mhy666@mail.dlut.edu.cn

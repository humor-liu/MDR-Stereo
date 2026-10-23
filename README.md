# MDR-Stereo
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

Multi-Domain Feature Enhancement and Region-Differentiated Perception for Stereo Matching
## Environment
* NVIDIA RTX 3090
* python 3.9
* Ubuntu
### Create a virtual environment and activate it.

```Shell
conda create -n MDR-Stereo python=3.9
conda activate MDR-Stereo
```
### Dependencies

```Shell
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.5.4

```
### Datasets 
The project requires the follow datasets:

<table style="border-collapse: collapse; width: 80%;">
  <tr>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html" target="_blank">SceneFlow</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo" target="_blank">KITTI-2012</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo" target="_blank">KITTI-2015</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://vision.middlebury.edu/stereo/submit3/" target="_blank">Middlebury</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://www.eth3d.net/datasets" target="_blank">ETH3D</a></td>
    <td style="border: 1px solid #000; padding: 8px;"><a class="custom-link" href="https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/" target="_blank">Virtual KITTI 2</a></td>
  </tr>
</table>
The datasets are organized as follows,

```
.
└── datasets
    ├── ETH3D
    │   ├── two_view_testing
    │   ├── two_view_training
    │   └── two_view_training_gt
    ├── KITTI12
    │   ├── testing
    │   └── training
    ├── KITTI15
    │   ├── testing
    │   └── training
    ├── Middlebury
    │   ├── 2005
    │   ├── 2006
    │   ├── 2014
    │   ├── 2021
    │   └── MiddEval3
    ├── SceneFlow
    │   ├── Driving
    │   ├── FlyingThings3D
    │   └── Monkaa
    └── VKITTI2 
        ├── Scene01
        ├── Scene02
        ├── Scene06
        ├── Scene18
        └── Scene20
```
# Evaluation

Download the pre-trained models
```
python evaluate_stereo.py
```
The pretrained models are available on [Google Drive](https://drive.google.com/drive/folders/15_G8zWTLGbRuquUV2ZeX3q_Q0YhuDlwq?usp=drive_link) and can be downloaded mamanually. 
The link includes the Depth Anything V2 weights, as well as the weights for SceneFlow and KITTI.

# Training 

Download Depthanythingv2 weights [DAV2](https://drive.google.com/file/d/1hUDjC2FWpJE9KBrZLLsFTNEWF2MFfiN8/view?usp=drive_link) 
```
Download the weights and place them in the root directory of the MDR-Stereo project. 
```

Train on SceneFlow
```
python train_stereo.py
```

# Acknowledgements

The project is based on [IGEV-Stereo](https://github.com/gangweix/IGEV) and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and we sincerely acknowledge their authors for opensourcing the excellent work. Besides, we would like to thank the CVPR reviewers and AC for their valuable feedback and recognition of our work.

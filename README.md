# MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation [CVPR 2022]

<p align="center"><img src="figure/pipline.jpg" width="100%" alt="" /></p>

> [**MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation**](https://arxiv.org/pdf/2111.12707),            
> Wenhao Li, Hong Liu, Hao Tang, Pichao Wang, Luc Van Gool,        
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2022*

<p float="left">
  <img src="figure/skating.gif" width="49%" />
  <img src="figure/anime.gif" width="49%" />


## Updates
- 03/24/2022: Demo and in-the-wild inference code is released!

## Installation

- Create a conda environment: ```conda create -n mhformer python=3.6```
- Download cudatoolkit=11.0 from [here](https://developer.nvidia.com/cuda-11.0-download-archive) and install 
- ```pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```
- ```pip install -r requirements.txt```
  
## Dataset setup

Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 
Or you can download the processed data from [here](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC?usp=sharing). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Download pretrained model

The pretrained model can be found in [here](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing), please download it and put it in the './checkpoint' directory. 

## Test the model

To test on pretrained model on Human3.6M:

```bash
python main.py --test --reload --previous_dir 'checkpoint/pretrained'
```

Here, we compare our MHFormer with recent state-of-the-art methods on Human3.6M dataset. Evaluation metric is Mean Per Joint Position Error (MPJPE) in mm​. 


|   Models    |  MPJPE   |
| :---------: | :------: |
| VideoPose3D |   46.8   |
| PoseFormer  |   44.3   |
|  MHFormer   | **43.0** |


## Train the model

To train on Human3.6M:

```bash
python main.py
```

## Demo
First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. 
Then, you need to put your in-the-wild videos in the './demo/video' directory. 

Run the command below:
```bash
python demo/vis.py --video sample_video.mp4
```

Sample demo output:

<p align="center"><img src="figure/sample_video.gif" width="60%" alt="" /></p>


## Citation

If you find our work useful in your research, please consider citing:

    @inproceedings{li2022mhformer,
      title={MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation},
      author={Li, Wenhao and Liu, Hong and Tang, Hao and Wang, Pichao and Van Gool, Luc},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
    }

## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
## Licence

This project is licensed under the terms of the MIT license.

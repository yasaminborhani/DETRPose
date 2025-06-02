
<h2 align="center">
  DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
</h2>

<p align="center">
  <a href="https://github.com/SebastianJanampa/DETRPose/master/LICENSE">
        <img alt="colab" src="https://img.shields.io/badge/license-apache%202.0-blue?style=for-the-badge">
  </a>

  <a href="">
        <img alt="arxiv" src="https://img.shields.io/badge/-paper-gray?style=for-the-badge&logo=arxiv&labelColor=red">
  </a>
  
  <a href="">
        <img alt="colab" src="https://img.shields.io/badge/-colab-blue?style=for-the-badge&logo=googlecolab&logoColor=white&labelColor=%23daa204&color=yellow">
  </a>

  <a href=''>
      <img src='https://img.shields.io/badge/-SPACE-orange?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF5500&color=orange'>
   </a>
   
</p>

<p align="center">
    📄 This is the official implementation of the paper:
    <br>
    <a href="">DETRPose: Real-time end-to-end transformer model for multi-person pose estimation</a>
</p>

</p>


<p align="center">
Sebastian Janampa and Marios Pattichis
</p>

<p align="center">
The University of New Mexico
  <br>
Department of Electrical and Computer Engineering
</p>

DETRPose is the first real-time end-to-end transformer model for multi-person pose estimation, 
achieving outstanding results on the COCO and CrowdPose datasets. In this work, we propose a 
new denoising technique suitable for pose estimation that uses the Object Keypoint Similarity (OKS) 
metric to generate positive and negative queries. Additionally, we develop a new classification head 
and a new classification loss that are variations of the LQE head and the varifocal loss used in D-FINE.


## 🚀 Updates
- [x] **\[2025.06.02\]** Release DETRPose code and weights.

## 📝 TODO
- [ ] Collab demo
- [ ] Hugging Face Space Demo
- [ ] Paper
- [ ] Inference time on Tesla V100 with ONNX+TensorRT backend 

## Model Zoo
### COCO val2017
| Model  | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**DETRPose-N** | 57.2 | 81.7 | 61.4 | 64.4 | 87.9 | 4.1 M | | 9.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_n.py) | [57.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_n.pth) | 
**DETRPose-S** | 67.0 | 87.6 | 72.8 | 73.5 | 92.4 | 11.9 M | | 33.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_s.py) | [67.0](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s.pth) | 
**DETRPose-M** | 69.4 | 89.2 | 75.4 | 75.5 | 93.7 | 23.5 M | | 67.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_m.py) | [69.4](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m.pth) | 
**DETRPose-L** | 72.5 | 90.6 | 79.0 | 78.7 | 95.0 | 36.8 M | | 107.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_l.py) | [72.5](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l.pth) | 
**DETRPose-X** | 73.3 | 90.5 | 79.4 | 79.4 | 94.9 | 82.3 M | | 239.5 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_x.py) | [73.3](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_x.pth) | 

### COCO test-devl2017
| Model  | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**DETRPose-N** | 56.7 | 83.1 | 61.1 | 64.4 | 89.3 | 4.1 M | | 9.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_n.py) | [](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_n.pth) | 
**DETRPose-S** | 66.0 | 88.3 | 72.0 | 73.2 | 93.3 | 11.9 M | | 33.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_s.py) | [](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s.pth) | 
**DETRPose-M** | 68.4 | 90.1 | 74.8 | 75.1 | 94.4 | 23.5 M | | 67.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_m.py) | [](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m.pth) | 
**DETRPose-L** | 71.2 | 91.2 | 78.1 | 78.1 | 95.7 | 36.8 M | | 107.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_l.py) | [71.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l.pth) | 
**DETRPose-X** | 72.2 | 91.4 | 79.3 | 78.8 | 95.7 | 82.3 M | | 239.5 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_x.py) | [72.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_x.pth) | 

### CrowdPose test
| Model  | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>E</sup> | AP<sup>M</sup> | AP<sup>H</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**DETRPose-N** | 56.0 | 80.7 | 59.6 | 65.0 | 56.6 | 46.6 | 4.1 M | | 8.8 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_n_crowdpose.py) | [57.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_n_crowdpose.pth) | 
**DETRPose-S** | 67.4 | 88.6 | 72.9 | 74.7 | 68.1 | 59.3 | 11.9 M | | 31.3  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_s_crowdpose.py) | [67.0](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s_crowdpose.pth) | 
**DETRPose-M** | 72.0 | 91.0 | 77.8 | 78.6 | 72.6 | 64.5 | 23.4 M | | 64.9  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_m_crowdpose.py) | [69.4](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m_crowdpose.pth) | 
**DETRPose-L** | 73.3 | 91.6 | 79.4 | 79.5 | 74.0 | 66.1 | 36.8 M | | 103.5  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_l_crowdpose.py) | [72.5](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l_crowdpose.pth) | 
**DETRPose-X** | 75.1 | 92.1 | 81.3 | 81.3 | 75.7 | 68.1 | 82.3 M | | 232.3  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_x_crowdpose.py) | [73.3](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_x_crowdpose.pth) | 

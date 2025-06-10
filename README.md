
<h2 align="center">
  DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
</h2>

<p align="center">
  <a href="https://github.com/SebastianJanampa/DETRPose/blob/main/LICENSE">
        <img alt="colab" src="https://img.shields.io/badge/license-apache%202.0-blue?style=for-the-badge">
  </a>

  <a href="">
        <img alt="arxiv" src="https://img.shields.io/badge/-paper-gray?style=for-the-badge&logo=arxiv&labelColor=red">
  </a>
  
  <a href="https://colab.research.google.com/github/SebastianJanampa/DETRPose/blob/main/DETRPose_tutorial.ipynb">
        <img alt="colab" src="https://img.shields.io/badge/-colab-blue?style=for-the-badge&logo=googlecolab&logoColor=white&labelColor=%23daa204&color=yellow">
  </a>

  <a href="https://huggingface.co/spaces/SebasJanampa/DETRPose">
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
- [x] **\[2025.06.04\]** Release [Google Colab Notebook](https://colab.research.google.com/github/SebastianJanampa/DETRPose/blob/main/DETRPose_tutorial.ipynb).
- [x] **\[2025.06.04\]** Release [HuggingFace 🤗 Space](https://huggingface.co/spaces/SebasJanampa/DETRPose).

## 📝 TODO
- [x] Collab demo
- [x] Hugging Face Space Demo
- [ ] Paper

## Model Zoo
### COCO val2017
| Model  | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**DETRPose-N** | 57.2 | 81.7 | 61.4 | 64.4 | 87.9 | 4.1 M | 2.80 ms | 9.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_n.py) | [57.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_n.pth) | 
**DETRPose-S** | 67.0 | 87.6 | 72.8 | 73.5 | 92.4 | 11.9 M | 4.99 ms | 33.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_s.py) | [67.0](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s.pth) | 
**DETRPose-M** | 69.4 | 89.2 | 75.4 | 75.5 | 93.7 | 23.5 M | 7.01 ms | 67.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_m.py) | [69.4](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m.pth) | 
**DETRPose-L** | 72.5 | 90.6 | 79.0 | 78.7 | 95.0 | 36.8 M | 9.50 ms | 107.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_l.py) | [72.5](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l.pth) | 
**DETRPose-X** | 73.3 | 90.5 | 79.4 | 79.4 | 94.9 | 82.3 M | 13.31 ms | 239.5 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_x.py) | [73.3](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_x.pth) | 

### COCO test-dev2017
| Model  | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**DETRPose-N** | 56.7 | 83.1 | 61.1 | 64.4 | 89.3 | 4.1 M | 2.80 ms | 9.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_n.py) | [56.7](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_n.pth) | 
**DETRPose-S** | 66.0 | 88.3 | 72.0 | 73.2 | 93.3 | 11.9 M | 4.99 ms | 33.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_s.py) | [66.0](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s.pth) | 
**DETRPose-M** | 68.4 | 90.1 | 74.8 | 75.1 | 94.4 | 23.5 M | 7.01 ms | 67.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_m.py) | [88.3](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m.pth) | 
**DETRPose-L** | 71.2 | 91.2 | 78.1 | 78.1 | 95.7 | 36.8 M | 9.50 ms | 107.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_l.py) | [71.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l.pth) | 
**DETRPose-X** | 72.2 | 91.4 | 79.3 | 78.8 | 95.7 | 82.3 M | 13.31 ms | 239.5 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_x.py) | [72.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_x.pth) | 

### CrowdPose test
| Model  | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>E</sup> | AP<sup>M</sup> | AP<sup>H</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**DETRPose-N** | 56.0 | 80.7 | 59.6 | 65.0 | 56.6 | 46.6 | 4.1 M | 2.72 ms | 8.8 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_n_crowdpose.py) | [57.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_n_crowdpose.pth) | 
**DETRPose-S** | 67.4 | 88.6 | 72.9 | 74.7 | 68.1 | 59.3 | 11.9 M | 4.80 ms | 31.3  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_s_crowdpose.py) | [67.0](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s_crowdpose.pth) | 
**DETRPose-M** | 72.0 | 91.0 | 77.8 | 78.6 | 72.6 | 64.5 | 23.4 M | 6.86 ms | 64.9  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_m_crowdpose.py) | [69.4](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m_crowdpose.pth) | 
**DETRPose-L** | 73.3 | 91.6 | 79.4 | 79.5 | 74.0 | 66.1 | 36.8 M | 9.03 ms | 103.5  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_l_crowdpose.py) | [72.5](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l_crowdpose.pth) | 
**DETRPose-X** | 75.1 | 92.1 | 81.3 | 81.3 | 75.7 | 68.1 | 82.3 M | 13.01 ms | 232.3  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_x_crowdpose.py) | [73.3](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_x_crowdpose.pth) | 

**Notes:**
- **Latency** is evaluated on a single  Tesla V100 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT==8.6.3$.

## Quick start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SebastianJanampa/DETRPose/blob/main/DETRPose_tutorial.ipynb)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/SebasJanampa/DETRPose)

### Setup

```shell
conda create -n detrpose python=3.11.9
conda activate detrpose
pip install -r requirements.txt
```

### Data Preparation
Create a folder named `data` to store the datasets
```
configs
src
tools
data
  ├── COCO2017
    ├── train2017
    ├── val2017
    ├── test2017
    └── annotations
  └── crowdpose
    ├── images
    └── annotations

```

<details>
  <summary> COCO2017 dataset </summary>
  Download COCO2017 from their [website](https://cocodataset.org/#download)
</details>
<details>
  <summary> CrowdPose dataset </summary>
  Download Crowdpose from their [github](https://github.com/jeffffffli/CrowdPose), or use the following command
  
```shell
pip install gdown # to download files from google drive
mkdir crowdpose
cd crowdpose
gdown 1VprytECcLtU4tKP32SYi_7oDRbw7yUTL # images
gdown 1b3APtKpc43dx_5FxizbS-EWGvd-zl7Lb # crowdpose_train.json
gdown 18-IwNa6TOGQPE0RqGNjNY1cJOfNC7MXj # crowdpose_val.json
gdown 13xScmTWqO6Y6m_CjiQ-23ptgX9sC-J9I # crowdpose_trainval.json
gdown 1FUzRj-dPbL1OyBwcIX2BgFPEaY5Yrz7S # crowdpose_test.json
unzip images.zip
```
</details>

### Usage
<details open>
  <summary> COCO2017 dataset </summary>
  
1. Set Model
```shell
export model=l # n s m l x
```

2. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --pretrain dfine_${model}_obj365 
```
if you choose `model=n`, do
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_n.py --device cuda --amp --pretrain dfine_n_obj365 
```

3. Testing (COCO2017 val)
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --resume <PTH_FILE_PATH> --eval
```

4. Testing (COCO2017 test-dev)
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --resume <PTH_FILE_PATH> --test
```
After running the command. You'll get a file named `results.json`. Compress it and submit it to the [COCO competition website](https://codalab.lisn.upsaclay.fr/competitions/7403#learn_the_details)

5. Replicate results (optional)
```shell
# First, download the official weights
wget https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_${model}.pth

# Second, run evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --resume detrpose_hgnetv2_${model}.pth --eval
```
</details>

<details>
  <summary> CrowdPose dataset </summary>
  
1. Set Model
```shell
export model=l # n s m l x
```

2. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py --device cuda --amp --pretrain dfine_${model}_obj365 
```
if you choose `model=n`, do
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_n_crowdpose.py --device cuda --amp --pretrain dfine_n_obj365 
```

3. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py --device cuda --amp --resume <PTH_FILE_PATH> --eval
```

4. Replicate results (optional)
```shell
# First, download the official weights
wget https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_${model}_crowdpose.pth

# Second, run evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py --device cuda --amp --resume detrpose_hgnetv2_${model}_crowdpose.pth --eval
```
</details>

## Tools
<details>
<summary> Deployment </summary>

<!-- <summary>4. Export onnx </summary> -->
1. Setup
```shell
pip install -r tools/inference/requirements.txt
export model=l  # n s m l x
```

2. Export onnx
For COCO model
```shell
python tools/deployment/export_onnx.py --check -c configs/detrpose/detrpose_hgnetv2_${model}.py -r detrpose_hgnetv2_${model}.pth
```

For CrowdPose model
```shell
python tools/deployment/export_onnx.py --check -c configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py -r detrpose_hgnetv2_${model}_crowdpose.pth
```

3. Export [tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
For a specific file
```shell
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

or, for all files inside a folder
```shell
python tools/deployment/export_tensorrt.py
```

</details>

<details>
<summary> Inference (Visualization) </summary>


1. Setup
```shell
export model=l  # n s m l x
```


<!-- <summary>5. Inference </summary> -->
2. Inference (onnxruntime / tensorrt / torch)

Inference on images and videos is supported.

For a single file
```shell
# For COCO model
python tools/inference/onnx_inf.py --onnx detrpose_hgnetv2_${model}.onnx --input examples/example1.jpg --annotator COCO
python tools/inference/trt_inf.py --trt detrpose_hgnetv2_${model}.engine --input examples/example1.jpg --annotator COCO
python tools/inference/torch_inf.py -c configs/detrpose/detrpose_hgnetv2_${model}.py -r <checkpoint.pth> --input examples/example1.jpg --device cuda:0 

# For CrowdPose model
python tools/inference/onnx_inf.py --onnx detrpose_hgnetv2_${model}_crowdpose.onnx --input examples/example1.jpg --annotator CrowdPose
python tools/inference/trt_inf.py --trt detrpose_hgnetv2_${model}_crowdpose.engine --input examples/example1.jpg --annotator CrowdPose
python tools/inference/torch_inf.py -c configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py -r <checkpoint.pth> --input examples/example1.jpg --device cuda:0 
```

For a folder
```shell
# For COCO model
python tools/inference/onnx_inf.py --onnx detrpose_hgnetv2_${model}.onnx --input examples --annotator COCO
python tools/inference/trt_inf.py --trt detrpose_hgnetv2_${model}.engine --input examples --annotator COCO
python tools/inference/torch_inf.py -c configs/detrpose/detrpose_hgnetv2_${model}.py -r <checkpoint.pth> --input examples --device cuda:0 

# For CrowdPose model
python tools/inference/onnx_inf.py --onnx detrpose_hgnetv2_${model}_crowdpose.onnx --input examples --annotator CrowdPose
python tools/inference/trt_inf.py --trt detrpose_hgnetv2_${model}_crowdpose.engine --input examples --annotator CrowdPose
python tools/inference/torch_inf.py -c configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py -r <checkpoint.pth> --input examples --device cuda:0

```
</details>

<details>
<summary> Benchmark </summary>

1. Setup
```shell
pip install -r tools/benchmark/requirements.txt
export model=l  # n s m l
```

<!-- <summary>6. Benchmark </summary> -->
2. Model FLOPs, MACs, and Params
```shell
# For COCO model
python tools/benchmark/get_info.py --config configs/detrpose/detrpose_hgnetv2_${model}.py

# For COCO model
python tools/benchmark/get_info.py --config configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py
```

3. TensorRT Latency
```shell
python tools/benchmark/trt_benchmark.py --infer_dir ./data/COCO2017/val2017 --engine_dir trt_engines
```

4. Pytorch Latency
```shell
# For COCO model
python tools/benchmark/torch_benchmark.py -c ./configs/detrpose/detrpose_hgnetv2_${model}.py --resume detrpose_hgnetv2_${model}.pth --infer_dir ./data/COCO/val2017

# For CrowdPose model
python tools/benchmark/torch_benchmark.py -c ./configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py --resume detrpose_hgnetv2_${model}_crowdpose.pth --infer_dir ./data/COCO/val2017
```
</details>


## Citation
If you use `DETRPose` or its methods in your work, please cite the following BibTeX entries:
<details open>
<summary> bibtex </summary>

```latex
TODO
```
</details>

## Acknowledgement
This work was supported in part by [Lambda.ai](https://lambda.ai).

Our work is built upon [D-FINE](https://github.com/Peterande/D-FINE), [Detectron2](https://github.com/facebookresearch/detectron2/tree/main), and [GroupPose](https://github.com/Michel-liu/GroupPose/tree/main).

✨ Feel free to contribute and reach out if you have any questions! ✨

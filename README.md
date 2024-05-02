# APFP
The code of APFP

# Introduction
In this repo, we provide the implementation of the following paper:
Adaptive Prototype Few-Shot Image Classification Method Based on Feature Pyramid

In this paper, a novel feature extraction method termed FResNet is introduced, which leverages feature pyramid structures to retain finer details during computation, resulting in feature maps with enhanced detailed features. Addressing the issue of utilizing sample mean for class prototypes in ProtoNet, we proposed a novel class prototype computation method called Adaptive Prototype. The Adaptive Prototype method adaptively computes optimal support set class prototypes based on the similarity between each support set sample and the query sample, yielding prototypes more aligned with the query sample features. Finally, the APFP method proposed in this paper was evaluated on the MiniImagenet and CUB datasets, demonstrating significant improvements compared to previous methods, achieving state-of-the-art performance on both datasets.

# Start
## Dataset
miniImageNet: Download Link: [BaiduCloud] [GoogleDrive].
CUB: Download Link: [BaiduCloud] [GoogleDrive].

Download the Mini-ImageNet dataset and the CUB dataset. Set the dataset paths in the `run_test.sh` script.

## Test

Mini-Imagnet test: 
```shell
cd script/mini-image/
./run_test.sh
```

cub test:
```shell
cd script/cub/
./run_test.sh
```
# Implementation environment
Note that the test accuracy may slightly vary with different Pytorch/CUDA versions, GPUs, etc.

Linux
Python 3.8.
torch: 1.11.0+cu113
GPU (RTX3090) + CUDA11.3.109

# Acknowledgments
Our code builds upon the the following code publicly available:
[https://github.com/Fei-Long121/DeepBDC]

# APFP
The code of APFP

# introduction
In this repo, we provide the implementation of the following paper:
Adaptive Prototype Few-Shot Image Classification Method Based on Feature Pyramid

In this paper, a novel feature extraction method termed FResNet is introduced, which leverages feature pyramid structures to retain finer details during computation, resulting in feature maps with enhanced detailed features. Addressing the issue of utilizing sample mean for class prototypes in ProtoNet, we proposed a novel class prototype computation method called Adaptive Prototype. The Adaptive Prototype method adaptively computes optimal support set class prototypes based on the similarity between each support set sample and the query sample, yielding prototypes more aligned with the query sample features. Finally, the APFP method proposed in this paper was evaluated on the MiniImagenet and CUB datasets, demonstrating significant improvements compared to previous methods, achieving state-of-the-art performance on both datasets.

# Acknowledgments
Our code builds upon the the following code publicly available:
[https://github.com/Fei-Long121/DeepBDC]

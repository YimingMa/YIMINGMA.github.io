---
key: 2021_10_08_01
title: Introduction to MobileNetV1
tags: ["Computer Vision", "Image Classifiers", "MobileNets"]
mathjax: true
mathjax_autoNumber: true
author: Yiming
comment: false
pageview: false
aside:
    toc: true
---

## Introduction

MobileNetV1 was proposed by Howard, Andrew G., et al. in [_MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications_](https://arxiv.org/abs/1704.04861) in 2017. In this paper, **depthwise separable convolution** is used to replace standard convolution to reduce computation. Although MobileNetV1 is smaller than other families of image classifiers, such as VGGs and Inceptions, it can still achieve comparable results on [ImageNet](https://www.image-net.org/).

## Standard Convolution vs. Depthwise Separable Convolution

### Settings

Suppose we are interested in transforming an input feature map $\boldsymbol{F}$ with the shape $D_F \times D_F \times M$ into an output feature map $\boldsymbol{G}$  with the shape $D_G \times D_G \times N$, where

- $D_F$ is the spatial width and height of $\boldsymbol{F}$;
- $M$ is the number of input channels (input depth);
- $D_G$ is the spatial width and height of $\boldsymbol{G}$;
- $N$ is the number of output channels (output depth).

### Standard Convolution

The **standard convolutional layer** is parametrized by the convolutional kernel $\boldsymbol{K}$: $D_K \times D_K \times M \times N$. Assume the padding is `"valid"` and the stride taken is `1`, then

$$
\label{eqn1}
\boldsymbol{G}_{k, \, l, \, n} = \sum_{i, \, j, \, m} \boldsymbol{K}_{i, \, j, \, m} \cdot \boldsymbol{F}_{k+i-1, \, l+j-1, \, m}.
$$

The number of multiplications involved in computing $\boldsymbol{G}$ via $\eqref{eqn1}$ is

$$
\label{eqn2}
D_K \cdot D_K \cdot M \cdot N \cdot D_G \cdot D_G.
$$

<style>
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
</style>

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/standard_convolution_filters.png" alt="Filters of Standard Convolution" class="center">

### Depthwise Separable Convolution

A **depthwise separable convolutional block** consists of two operations – a **depthwise convolution** and a **pointwise convolution**.

#### Depthwise Convolution

Let $\hat{\boldsymbol{K}}$: $D_K \times D_K \times M$ be the **depthwise convolutional kernel** with one filter per channel and $\hat{\boldsymbol{G}}$: $D_G \times D_G \times M$ be the corresponding output feature map. Then

$$
\label{eqn3}
\hat{\boldsymbol{G}}_{k, \, l, \, m} = \sum_{i, \, j} \hat{\boldsymbol{K}}_{i, \, j, \, m} \cdot \boldsymbol{F}_{k+i-1, \, l+j-1, \, m},
$$

taking

$$
\label{eqn4}
D_K \cdot D_K \cdot M \cdot D_G \cdot D_G
$$

multiplications.

<u>Notice</u>: Depthwise convolution only filters input channels, so it does not combine them to create new features.

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/depthwise_convolution_filters.png" alt="Filters of Depthwise Convolution" class="center">

#### Pointwise Convolution

Let $\tilde{\boldsymbol{K}}$: $1 \times 1 \times M \times N$ be the **pointwise convolutional kernel** which takes $\hat{\boldsymbol{G}}$: $D_G \times D_G \times M$ as input and outputs $\tilde{\boldsymbol{G}}$: $D_G \times D_G \times N$. Then

$$
\label{eqn5}
\tilde{\boldsymbol{G}}_{k, \, l, \, n} = \sum_{m} \tilde{\boldsymbol{K}}_{1, \, 1,\, m, \, n} \cdot \hat{G}_{k, \, l, \, m},
$$

which entails multiplications of

$$
\label{eqn6}
M \cdot N \cdot D_G \cdot D_G
$$

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/pointwise_convolution_filters.png" alt="Fliters of Pointwise Convolution" class="center">

Thus, the overall **depthwise separable convolution** can be expressed as

$$
\label{eqn7}
\tilde{\boldsymbol{G}}_{k, \, l, \, n} = \sum_{m} \tilde{\boldsymbol{K}}_{1, \, 1, \, m, \, n} \cdot \sum_{i, \, j} \hat{\boldsymbol{K}}_{i, \, j, \, m} \cdot \boldsymbol{F}_{k + i - 1, \, l + j - 1, \, m}.
$$

The number of multiplications in $\eqref{eqn7}$ is

$$
\label{eqn8}
D_K \cdot D_K \cdot M \cdot D_G \cdot D_G + M \cdot N \cdot D_G \cdot D_G,
$$

which is

$$
\frac{1}{N} + \frac{1}{D_K^2} \notag
$$

of the standard convolution.

### Examples

#### Standard Convolution

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/standard_convolution.png" alt="Standard Convolution" class="center">

#### Depthwise Separable Convolution

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/depthwise_convolution.png"
alt="Depthwise Convolution" class="center">

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/pointwise_convolution.png" alt="Pointwise Convolution" class="center">

## The Architecture of MobileNetV1

Except that the first layer is a standard convolution, all other convolutions in MobileNetV1 are depthwise separable. Batch normalization and ReLU activation are also used after each convolution.

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/standard_conv_layer_vs_depthwise_separable_conv_layer.png"
alt="Depthwise Convolution" class="center">

The structure of MobileNetV1 is shown below.

| Type / Stride 	                | Filter Shape 	               | Input Size 	                |
|---------------------------------|------------------------------|------------------------------|
| Conv / s2     	                | 3×3×3×32     	               | 224×224×3                    |
| Conv dw / s1                    | 3×3×32 dw                    | 112×112×32                   |
| Conv / s1                       | 1×1×32×64                    | 112×112×32                   |
| Conv dw / s2                    | 3×3×64 dw                    | 112×112×64                   |
| Conv / s1                       | 1×1×64×128                   | 56×56×64                     |
| Conv dw / s1                    | 3×3×128 dw                   | 56×56×128                    |
| Conv / s1                       | 1×1×128×128                  | 56×56×128                    |
| Conv dw / s2                    | 3×3×128 dw                   | 56×56×128                    |
| Conv / s1                       | 1×1×128×256                  | 28×28×128                    |
| Conv dw / s1                    | 3×3×256 dw                   | 28×28×256                    |
| Conv / s1                       | 1×1×256×256                  | 28×28×256                    |
| Conv dw / s2                    | 3×3×256 dw                   | 28×28×256                    |
| Conv / s1                       | 1×1×256×512                  | 14×14×256                    |
| 5 × (Conv dw / s1 + Conv / s2)  | (3×3×512 dw) + (1×1×512×512) | (14×14×512) + (14×14×512)    |
| Conv dw / s2                    | 3×3×512 dw                   | 14×14×512                    |
| Conv / s1                       | 1×1×512×1024                 | 7×7×512                      |
| Conv dw / s2                    | 3×3×1024 dw                  | 7×7×1024                     |
| Conv / s1                       | 1×1×1024×1024                | 7×7×1024                     |
| AvgPool / s1                    | 7×7                          | 7×7×1024                     |
| FC / s1                         | 1024×1000                    | 1×1×1024                     |
| Softmax / s1                    | Classifier                   | 1×1×1000                     |

## Experiments

### Results on [ImageNet](https://www.image-net.org/)

Results from the original paper [_MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications_](https://arxiv.org/abs/1704.04861):

| Model                                           | ImageNet Accuracy | Mult-Adds (M) | Parameters (M) |
| ----------------------------------------------- | ----------------- | ------------- | -------------- |
| [MobileNetV1](https://arxiv.org/abs/1704.04861) | 70.6%             | 569           | 4.2            |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)    | 69.8%             | 1550          | 6.8            |
| [VGG16](https://arxiv.org/abs/1409.1556)        | 71.5%             | 15300         | 138            |

Results on [Keras Applications](https://keras.io/api/applications/):

| Model                                            | Size (MB)         | Top-1 Accuracy | Top-5 Accuracy | Parameters  | Depth | Time (ms) per inference step (CPU) | Time (ms) per inference step (GPU) |
|--------------------------------------------------|-------------------|----------------|----------------|-------------|-------|------------------------------------|------------------------------------|
| [Xception](https://arxiv.org/abs/1610.02357)     | 88                | 0.790          | 0.945          | 22,910,480  | 126   | 109.42                             | 8.06                               |
| [VGG16](https://arxiv.org/abs/1409.1556)         | 528               | 0.713          | 0.901          | 138,357,544 | 23    | 69.50                              | 4.16                               |
| [VGG19](https://arxiv.org/abs/1409.1556)         | 549               | 0.713          | 0.900          | 143,667,240 | 26    | 84.75                              | 4.38                               |
| [ResNet50](https://arxiv.org/abs/1512.03385)     | 98                | 0.749          | 0.921          | 25,636,712  | -     | 58.20                              | 4.55                               |
| [ResNet101](https://arxiv.org/abs/1512.03385)    | 171               | 0.764          | 0.928          | 44,707,176 | -     | 89.59                              | 5.19                               |
| [InceptionV3](https://arxiv.org/abs/1512.00567)  | 92                | 0.779          | 0.937          | 23,851,784  | 159   | 42.25                              | 6.86                               |
| [MobileNetV1](https://arxiv.org/abs/1704.04861)  | 16                | 0.704          | 0.895          | 4,253,864   | 88    | 22.60                              | 3.44                               |

- The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.
- Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.
- Time per inference step is the average of 30 batchs and 10 repetitions.
  - CPU: AMD EPYC Processor (with IBPB) (92 core)
  - Ram: 1.7T
  - GPU: Tesla A100
  - Batch size: 32
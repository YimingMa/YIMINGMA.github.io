---
key: 2021_10_08_01
title: Introduction to MobileNet-V1
tags: ["Computer Vision", "Image Classifiers", "MobileNets"]
mathjax: true
author: Yiming
comment: false
pageview: false
aside:
    toc: true
---

## Introduction

MobileNet-V1 was proposed by Howard, Andrew G., et al. in [_MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications_]([[1704.04861\] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (arxiv.org)](https://arxiv.org/abs/1704.04861)) in 2017. In this paper, **depthwise separable convolution** is used to replace standard convolution to reduce computation. Although MobileNet-V1 is smaller than other families of image classifiers, such as VGGs and Inceptions, it can still achieve comparable results on [ImageNet]([ImageNet (image-net.org)](https://www.image-net.org/)).

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
\boldsymbol{G}_{k, \, l, \, n} = \sum_{i, \, j, \, m} \boldsymbol{K}_{i, \, j, \, m} \cdot \boldsymbol{F}_{k+i-1, \, l+j-1, \, m}.
$$
The number of multiplications involved in computing $\boldsymbol{G}$ is
$$
D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F.
$$
![Standard Convolution](\posts.assets\2021-10-08-introduction-to-MobileNet-V1.assets\standard_convolution_filters.png)

### Depthwise Separable Convolution

A **depthwise separable convolutional block** consists of two operations – a **depthwise convolution** and a **pointwise convolution**.

#### Depthwise Convolution


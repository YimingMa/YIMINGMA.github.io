---
key: 2021_10_08_01
title: Introduction to MobileNet-V1
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
\label{eqn1}
\boldsymbol{G}_{k, \, l, \, n} = \sum_{i, \, j, \, m} \boldsymbol{K}_{i, \, j, \, m} \cdot \boldsymbol{F}_{k+i-1, \, l+j-1, \, m}.
$$

The number of multiplications involved in computing $\boldsymbol{G}$ via $\eqref{eqn1}$ is

$$
\label{eqn2}
D_K \cdot D_K \cdot M \cdot N \cdot D_G \cdot D_G.
$$

![Standard Convolution](/posts.assets/2021-10-08-introduction-to-MobileNet-V1.assets/standard_convolution_filters.png)

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

![Depthwise Convolution](/posts.assets/2021-10-08-introduction-to-MobileNet-V1.assets/depthwise_convolution_filters.png)

### Pointwise Convolution

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

Thus, the overall **depthwise separable convolution** can be expressed as

$$
\label{eqn7}
\tilde{\boldsymbol{G}}_{k, \, l, \, n} = \sum_{m} \tilde{\boldsymbol{K}}_{1, \, 1, \, m, \, n} \cdot \sum_{i, \, j} \hat{\boldsymbol{K}}_{i, \, j, \, m} \cdot \boldsymbol{F}_{k + i - 1, \, l + j - 1, \, m}.
$$

The number of multiplications in $\eqref{eqn7}$ is

$$
\label{eqn8}
D_K \cdot D_k \cdot M \cdot D_G \cdot D_G + M \cdot N \cdot D_G \cdot D_G,
$$

which is $\frac{1}{N} + \frac{1}{D_K^2}$ of the standard convolution.

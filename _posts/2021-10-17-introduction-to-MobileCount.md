---
key: 2021_10_17_01
title: Introduction to MobileCount
tags: ["Computer Vision", "Crowd Counting"]
mathjax: true
mathjax_autoNumber: true
author: Yiming
comment: false
pageview: false
aside:
    toc: true
---

<style>
.center1 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 10%;
}
</style>

<style>
.center2 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 20%;
}
</style>

<style>
.center3 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 30%;
}
</style>

<style>
.center4 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 40%;
}
</style>

<style>
.center5 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
</style>

<style>
.center6 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 60%;
}
</style>

<style>
.center7 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 70%;
}
</style>

<style>
.center8 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 80%;
}
</style>

<style>
.center9 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 90%;
}
</style>

## Introduction

In [*MobileCount: An efficient encoder-decoder framework for real-time crowd counting*](https://www.sciencedirect.com/science/article/abs/pii/S0925231220308912), the authors proposed a light-weight crowd counting model – MobileCount, whose structure follows an autoencoder pattern. It uses [MobileNetV2](https://arxiv.org/abs/1801.04381) as the encoder and [RefineNet](https://arxiv.org/abs/1611.06612) as the decoder. A multi-layer knowledge distillation method was also employed to improve its performance without increasing computation.

## The Architecture of MobileCount

The image below shows the structure of MobileCount. Its encoder is adapted from [MobileNetV2](https://yimingma.github.io/2021/10/08/introduction-to-MobileNetV2.html) by reducing the number of inverted residual blocks from 7 to 4. The authors claimed that this reduction can improve the performance while decrease the number of FLOPs. And to make the model even lighter, a 3×3 max pooling layer with the stride of 2 is added before this encoder part to reduce the input resolution. 

<img src="/posts.assets/2021-10-17-introduction-to-MobileCount.assets/architecture_of_mobile_count.png" alt="The Structure of MobileCount" class="center9">

As for the decoder component of MobileCount, [lightweight RefineNet](https://arxiv.org/abs/1810.03272) which was originally designed for semantic segmentation is exploited. The decoding process starts with the propagation of the last output from the backbone through **chained residual pooling (CRP)** blocks (b) before being fed into fusion blocks (c). Inside the fusion block, input feature maps with both high and low resolutions are convolved with 1×1 convolutional kernels first. Then the low resolution feature map is upsampled to match the high resolution, and then these two feature maps will be summed up.

Regarding the prediction layer, 1×1 convolution will again be utilized to reduce depth of the feature map, and then the resulting feature map will be upsampled to the original input image size by bilinear interpolation.

## Loss Functions

The ground-truth dot annotations are smoothed with Gaussian kernels first, and the pixel-wise mean squared loss between density maps are used for training. Let $$\boldsymbol{D}^{\text{GT}}_i$$ be the $i$-th ground-truth density map and $$\boldsymbol{D}^{\text{Pred}}_i$$ be its prediction. Then the loss function for training is defined as

$$
L(\boldsymbol{\mathcal{D}}^{\text{GT}}, \, \boldsymbol{\mathcal{D}}^{\text{Pred}}) := \frac{1}{N} \sum_{i=1}^N \| \boldsymbol{D}^{\text{GT}}_i - \boldsymbol{D}^{\text{Pred}}_i \|_2^2. \notag
$$

For evaluation, the mean absolute error and the mean squared error between the ground-truth total count and the predicted total count are used:

$$
\begin{align*}
\text{MAE} (\boldsymbol{C}^{\text{GT}}, \, \boldsymbol{C}^{\text{Pred}}) := & \frac{1}{N} \sum_{i=1}^N \left| C^\text{GT}_i - C^\text{Pred}_i \right| \\
\text{MSE} (\boldsymbol{C}^{\text{GT}}, \, \boldsymbol{C}^{\text{Pred}}) := \sqrt{ \frac{1}{N} \sum_{i=1}^N \left( C^\text{GT}_i - C^\text{Pred}_i \right)^2}, \notag
\end{align*}
$$

where $$C^\text{GT}_i$$ is the $i$-th ground-truth total count and $$C^\text{Pred}_i$$ is its estimation. 

## Experiments


---
key: 2021_10_11_01
title: Introduction to MobileNetV3
tags: ["Computer Vision", "Image Classifiers", "MobileNets"]
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
  width: 40%;
}
</style>

<style>
.center2 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
</style>

<style>
.center3 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 60%;
}
</style>

<style>
.center4 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 70%;
}
</style>

<style>
.center5 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 80%;
}
</style>

<style>
.center6 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 90%;
}
</style>

## Introduction

MobileNetV3, proposed in [*Searching for MobileNetV3*](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf), is based on the combination of [**hardaware-aware network architecture search** (**NAS**)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf) and [**NetAdapt**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.pdf). These two techniques originate from reinforcement learning, in which both the accuracy and the latency are considered during the design of the reward function. NAS is used to find the global network structures, with the starting point of [MnasNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf), while NetAdapt searches the best layer-wise structure by tuning the number of filters of each layer. This third version of MobileNet also introduces a lightweight attention mechanism – [**Squeeze-and-Excitation**](https://arxiv.org/abs/1709.01507), and a new activation function – **h-swish**, into the inverted residual block proposed in [*MobileNetV2: Inverted Residuals and Linear Bottlenecks*](https://ieeexplore.ieee.org/document/8578572). This post will concentrate only on the new residual block and the hard version of **swish**, while how NAS and NetAdapt work will NOT be explained.

## Inverted Residual Blocks with Squeeze-and-Excitation

<img src="/posts.assets/2021-10-11-introduction-to-MobileNetV3.assets/inverted_residual_block_with_se.png" alt="Standard Convolution" class="center6">

## Hard Swish

### Swish

A nonlinear activation called **swish** was introduced in [*Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning*](https://arxiv.org/abs/1702.03118), [*Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units*](https://openreview.net/forum?id=Bk0MRI5lg) and [*Searching for Activation Functions*](https://arxiv.org/abs/1710.05941) to replace ReLU, and the accuracy of the resulted neural network can be significantly improved. Swish is defined as

$$
\label{eqn1}
\text{swish}(x) := x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}.
$$

Since $\eqref{eqn1}$ involves a sigmoid function, computation is much expensive on mobile devices. Thus, authors of MobileNetV3 proposed **h-swish** by approximating sigmoid with its piecewise linear hard analog $$\frac{\text{ReLU}6(x+3)}{6}$$, which means

$$
\label{eqn2}
\text{h-swish}(x) := x \cdot \frac{\text{ReLU}6(x+3)}{6}.
$$

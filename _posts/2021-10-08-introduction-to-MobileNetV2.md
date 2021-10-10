---
key: 2021_10_08_02
title: Introduction to MobileNetV2
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

Researchers have found that depthwise kernels in [MobileNetV1](https://arxiv.org/abs/1704.04861) after training can become very sparse. This problem is caused by information loss after ReLU activation, so in [_MobileNetV2: Inverted Residuals and Linear Bottlenecks_](https://arxiv.org/abs/1801.04381), authors suggest that it is important to remove non-linearities in the narrow layers. A novel layer module – the **inverted residual block** has also been proposed, which takes a low-dimesional feature map as an input, and then it is expanded to be high-dimensional and subsequently filtered depthwise. At last, these filtered features are projected back to the low-dimensional space with a linear convolution, and then concatenated with the original input.


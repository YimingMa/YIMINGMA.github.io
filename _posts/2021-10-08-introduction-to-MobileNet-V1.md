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


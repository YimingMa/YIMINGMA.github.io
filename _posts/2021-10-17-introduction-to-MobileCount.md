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

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

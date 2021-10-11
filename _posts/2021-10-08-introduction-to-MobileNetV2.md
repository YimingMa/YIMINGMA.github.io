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

<style>
.center1 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
</style>

<style>
.center2 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 60%;
}
</style>

<style>
.center3 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 70%;
}
</style>

<style>
.center4 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 80%;
}
</style>

<style>
.center5 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 90%;
}
</style>

## Introduction

Researchers have found that depthwise kernels in [MobileNetV1](https://arxiv.org/abs/1704.04861) after training can become very sparse. This problem is caused by information loss after ReLU activation, so in [_MobileNetV2: Inverted Residuals and Linear Bottlenecks_](https://arxiv.org/abs/1801.04381), authors suggest that it is important to remove nonlinearity in the narrow layers. A novel layer module – the **inverted residual block** has also been proposed, which takes a low-dimensional feature map as an input, and then it is expanded to be high-dimensional and subsequently filtered depthwise. At last, these filtered features are projected back to the low-dimensional space with a linear convolution, and then concatenated with the original input.

## Linear Bottlenecks

### Limitations of Nonlinear Transformations

Consider a deep neural network with $n$ layers $$\{L^{(k)}\}_{k=1}^n$$, each of which has an output tensor of dimensions $$h^{(k)} \times w^{(k)} \times d^{(k)}$$. We treat each of them as a container of $h^{(k)} \cdot w^{(k)}$  "pixels" with $d^{(k)}$ dimensions. For a specific layer $$L^{(k)}$$, we assume a nonlinear transformation is taken after a convolutional operation. Namely, let $\boldsymbol{T}^{(k)} \in \mathbb{R}^{d^{(k)}\times d^{(k-1)}}$, and denote the output from the convolutional operation on the input tensor $\boldsymbol{a}^{(k-1)} \in \mathbb{R}^{d^{(k-1)}}$ as $\boldsymbol{z}^{(k)} \in \mathbb{R}^{d^{(k)}}$, i.e.,

$$
\label{eqn1}
\boldsymbol{z}^{(k)} := \boldsymbol{T}^{(k)} \boldsymbol{a}^{(k-1)}.
$$

Let $f^{(k)}$ be the activation function, so

$$
\label{eqn2}
\boldsymbol{a}^{(k)} = f^{(k)} \left( \boldsymbol{z}^{(k)} \right) = f^{(k)} \left( \boldsymbol{T}^{(k)} \boldsymbol{a}^{(k-1)} \right) \in \mathbb{R}^{d^{(k)}}.
$$

For an input set of real images $$\{\mathcal{I}_i\}_{i=1}^m$$ and any layer $$L^{(k)}$$, we say that the set of layer activations $$\{\boldsymbol{z}^{(k)}_i\}_{i=1}^m$$ forms a **manifold of interest**. We usually assume the batch size $m$ is no greater than the output dimension $d^{(k)}$, so this manifold of interest can be *imbedded* into a subspace of $\mathbb{R}^{d^{(k)}}$, denoted as $\mathbb{R}^{l}$ with $l \le m$.

At a first glance, such a fact could then be exploited to reduce the dimensionality of a layer, i.e. we can make $d_k = m$ in the ideal case to decrease the number of overall calculations. However, this intuition breaks down when we recall that deep convolutional neural networks actually nonlinear per coordinate transformations such as ReLU. To be specific, if $d^{(k)}$ is too small, i.e. $d^{(k)} \approx m$, then a significant fraction of information carried by features $$\{\boldsymbol{a}^{(k-1)}_i\}_{i=1}^m$$ will be lost after such nonlinear transformations. For illustration, assume these features $$\{\boldsymbol{a}^{(k-1)}_i\}_{i=1}^m \subset \mathbb{R}^{d^{(k-1)}}$$, which are the input of layer $L^{(k)}$, have a shape of spiral in $\mathbb{R}^2$ (i.e. $l=2$). Use a random matrix $\boldsymbol{T}^{(k)} \in \mathbb{R}^{ d^{(k)} \times d^{(k-1)} }$ to denote the convolutional operation, so

$$
\label{eqn3}
\boldsymbol{a}^{(k)}_i = \text{ReLU}(\boldsymbol{T}^{(k)}\boldsymbol{a}^{(k-1)}_i), \, i = 1, \, \cdots, \, m.
$$

We then try to transform each $\boldsymbol{a}^{(k)}_i$ back to $\boldsymbol{a}^{(k-1)}_i$ using the (pseudo) inverse of $\boldsymbol{T}^{(k)}$ to evaluate the information loss from the ReLU operation in $\eqref{eqn3}$. Namely,

$$
\label{eqn4}
\hat{\boldsymbol{a}}^{(k-1)}_i := \left(\boldsymbol{T}^{(k)}\right)^{-1}\boldsymbol{a}^{(k)}_i = \left(\boldsymbol{T}^{(k)}\right)^{-1}\text{ReLU}(\boldsymbol{T}^{(k)}\boldsymbol{a}^{(k-1)}_i).
$$

The images below illustrate how the amount of information loss varies with $d^{(k)}$ (with $d^{(k)}$ set to be 2, 3, 5, 15 and 30) by comparing $$\{ \boldsymbol{a}^{(k-1)}_i \}_{i=1}^m$$ and $$\{ \hat{\boldsymbol{a}}^{(k-1)}_i \}_{i=1}^m$$.

![Effects of ReLU](/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/effects_of_relu.png)

**Conclusion**: we should use nonlinear transformation in layer $L^{(k)}$ *only* when $d^{(k)}$ is significantly larger than $l$. In other words,

- we can include an expansion layer to increase the number of channels;
- or we don't use activations for layers with small output dimensions.

### The Structure of Linear Bottlenecks

#### Standard Convolution

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/standard_convolution.png" alt="Standard Convolution" class="center1">

#### Depthwise Separable Convolution

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/depthwise_separable_convolution.png" alt="Depthwise Separable Convolution" class="center2">

#### Depthwise Separable Convolution with Linear Bottleneck

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/depthwise_separable_convolution_with_linear_bottleneck.png" alt="Depthwise Separable Convolution with Linear Bottleneck" class="center3">

#### Bottleneck with Expansion Layer

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/bottleneck_with_expansion_layer.png" alt="Bottleneck with Expansion Layer" class="center3">
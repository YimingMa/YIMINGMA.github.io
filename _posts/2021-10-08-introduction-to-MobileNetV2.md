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

1

## Introduction

Researchers have found that depthwise kernels in [MobileNetV1](https://arxiv.org/abs/1704.04861) after training can become very sparse. This problem is caused by information loss after ReLU activation, so in [_MobileNetV2: Inverted Residuals and Linear Bottlenecks_](https://arxiv.org/abs/1801.04381), authors suggest that it is important to remove non-linearities in the narrow layers. A novel layer module – the **inverted residual block** has also been proposed, which takes a low-dimensional feature map as an input, and then it is expanded to be high-dimensional and subsequently filtered depthwise. At last, these filtered features are projected back to the low-dimensional space with a linear convolution, and then concatenated with the original input.

## Linear Bottlenecks

Consider a deep neural network with $n$ layers $\left\{L^{(k)}\right\}_{k=1}^n$, each of which has an output tensor of dimensions $h^{(k)} \times w^{(k)} \times d^{(k)}$. We treat each of them as a container of $h^{(k)} \cdot w^{(k)}$  "pixels" with $d^{(k)}$ dimensions. For a specific layer $k,$ we assume a non-linear transformation is taken after a convolutional operation. Namely, let $\boldsymbol{T}^{(k)} \in \mathbb{R}^{d^{(k)}\times d^{(k-1)}}$, and denote the output from the convolutional operation on the input tensor $\boldsymbol{a}^{(k-1)} \in \mathbb{R}^{d^{(k-1)}}$ as $\boldsymbol{z}^{(k)} \in \mathbb{R}^{d^{(k)}}$, i.e.,

$$
\label{eqn1}
\boldsymbol{z}^{(k)} := \boldsymbol{T}^{(k)} \boldsymbol{a}^{(k-1)}.
$$

Let $f^{(k)}$ be the activation function, so

$$
\label{eqn2}
\boldsymbol{a}^{(k)} = f^{(k)} \left( \boldsymbol{z}^{(k)} \right) = f^{(k)} \left( \boldsymbol{T}^{(k)} \boldsymbol{a}^{(k-1)} \right) \in \mathbb{R}^{d^{(k)}}.
$$

For an input set of real images $\{\mathcal{I}_i\}_{i=1}^m$ and any layer $L^{(k)}$, we say that the set of layer activations $\\{\boldsymbol{z}^{(k)}_i\\}_{i=1}^m$ forms a **manifold of interest**. We usually assume the batch size $m$ is no greater than the output dimension $d^{(k)}$, so this manifold of interest can be *imbedded* into a subspace of $\mathbb{R}^{d^{(k)}}$, denoted as $\mathbb{R}^{l}$ with $l \le m$.

At a first glance, such a fact could then be exploited to reduce the dimensionality of a layer, i.e. we can make $d_k = m$ in the ideal case to decrease the number of overall calculations. However, this intuition breaks down when we recall that deep convolutional neural networks actually non-linear per coordinate transformations such as ReLU. To be specific, if $d^{(k)}$ is too small, i.e. $d^{(k)} \approx m$, then a significant fraction of information carried by features $\\{\boldsymbol{a}^{(k-1)}_i\\}_{i=1}^m$ will lost after such non-linear transformations. For illustration, assume these features $\\{\boldsymbol{a}^{(k-1)}_i\\}_{i=1}^m \subset \mathbb{R}^{d^{(k-1)}}$, which are the input of layer $L^{(k)}$, have a shape of spiral in $\mathbb{R}^2$. Use a random matrix $\boldsymbol{T}^{(k)} \in \mathbb{R}^{ d^{(k)} \times d^{(k-1)} }$ ($d^{(k)}$ is set to be 2, 3, 5, 15 and 30) to denote the convolutional operation, so

$$
\label{eqn3}
\boldsymbol{a}^{(k)}_i = \text{ReLU}(\boldsymbol{T}^{(k)}\boldsymbol{a}^{(k-1)}_i), \, i = 1, \, \cdots, \, m.
$$

We then try to transform each $\boldsymbol{a}^{(k)}_i$ back to $\boldsymbol{a}^{(k-1)}_i$ using the (pseudo) inverse of $\boldsymbol{T}^{(k)}$ to evaluate the information loss from the ReLU operation in $\eqref{eqn3}$. Namely,

$$
\label{eqn4}
\hat{\boldsymbol{a}}^{(k-1)}_i := \boldsymbol{T}^{-1}\boldsymbol{a}^{(k)}_i = \boldsymbol{T}^{-1}\text{ReLU}(\boldsymbol{T}\boldsymbol{a}^{(k-1)}_i).
$$

The image below illustrate how the amount of information varies with $d^{(k)}$.

<style>
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
</style>

![Effects of ReLU](/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/effects_of_relu.png)

---
key: 2021_10_08_01
title: Introduction to MobileNetV1
layout: article
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

MobileNetV1 was proposed by Howard, Andrew G., et al. in [*MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*](https://arxiv.org/abs/1704.04861) in 2017. In this paper, **depthwise separable convolution** is used to replace standard convolution to reduce computation. Although MobileNetV1 is smaller than other families of image classifiers, such as VGGs and Inceptions, it can still achieve comparable results on [ImageNet](https://www.image-net.org/).

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

![Filters of standard convolution.](/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/standard_convolution_filters.png "Filters of standard convolution."){:width="50%"; title="Filters of standard convolution.";}

For example, let $$\boldsymbol{F} \in \mathbb{R}^{3 \times 3 \times 3}$$, and its channelwise elements are given by

$$
\label{eqn3}
\boldsymbol{F}_{:, \, :, \, 1} =
\begin{bmatrix}
    0 & 3 & 2 \\
    0 & 4 & 1 \\
    1 & 0 & 1
\end{bmatrix}, \,
\boldsymbol{F}_{:, \, :, \, 2} =
\begin{bmatrix}
    0 & 1 & 4 \\
    2 & 4 & 2 \\
    1 & 3 & 4
\end{bmatrix}, \,
\boldsymbol{F}_{:, \, :, \, 3} =
\begin{bmatrix}
    4 & 4 & 0 \\
    2 & 1 & 1 \\
    1 & 4 & 3
\end{bmatrix}.
$$

Assume $$\boldsymbol{K} \in \mathbb{R}^{2 \times 2 \times 3 \times 2}$$, and the elements of its first filter are given by

$$
\boldsymbol{K}_{:, \, :, \, 1, \, 1} =
\begin{bmatrix}
    1 & 0 \\
    0 & 0
\end{bmatrix},
\boldsymbol{K}_{:, \, :, \, 2, \, 1} =
\begin{bmatrix}
    1 & 0 \\
    0 & 1
\end{bmatrix},
\boldsymbol{K}_{:, \, :, \, 3, \, 1} =
\begin{bmatrix}
    0 & 0 \\
    0 & 1
\end{bmatrix}, \notag
$$

and those of the second filter are given by

$$
\boldsymbol{K}_{:, \, :, \, 1, \, 2} =
\begin{bmatrix}
    0 & 1 \\
    0 & 0
\end{bmatrix},
\boldsymbol{K}_{:, \, :, \, 2, \, 2} =
\begin{bmatrix}
    0 & 1 \\
    1 & 0
\end{bmatrix},
\boldsymbol{K}_{:, \, :, \, 3, \, 2} =
\begin{bmatrix}
    0 & 0 \\
    1 & 0
\end{bmatrix}. \notag
$$

Using no paddings and the stride of 1, we know $$\boldsymbol{G} \in \mathbb{R}^{2 \times 2 \times 2}$$, and $$\boldsymbol{G}_{1, \, 1, \, 1}$$ is given by

$$
\begin{align*}
    & \boldsymbol{e}^\intercal \left( F_{1:\,2, \, 1:\,2, \, 1} \cdot \boldsymbol{K}_{:, \, :, \, 1, \, 1} + F_{1:\,2, \, 1:\,2, 2} \cdot \boldsymbol{K}_{:, \, :, \, 2, \, 1} + F_{1:\,2, \, 1:\,2, 3} \cdot \boldsymbol{K}_{:, \, :, \, 3, \, 1} \right) \boldsymbol{e} \\
    = & \boldsymbol{e}^\intercal \left(
        \begin{bmatrix}
            0 & 3 \\
            0 & 4
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            1 & 0 \\
            0 & 0
        \end{bmatrix}
        +
        \begin{bmatrix}
            0 & 1 \\
            2 & 4
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            1 & 0 \\
            0 & 1
        \end{bmatrix}
        +
        \begin{bmatrix}
            4 & 4 \\
            2 & 1
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            0 & 0 \\
            0 & 1
        \end{bmatrix}
    \right) \boldsymbol{e} \\
    = & \boldsymbol{e}^\intercal \left(
        \begin{bmatrix}
            0 & 0 \\
            0 & 0
        \end{bmatrix}
        +
        \begin{bmatrix}
            0 & 1 \\
            2 & 4
        \end{bmatrix}
        +
        \begin{bmatrix}
            0 & 4 \\
            0 & 1
        \end{bmatrix}
    \right) \boldsymbol{e} \\
    = & \begin{bmatrix}
            1 & 1
        \end{bmatrix}
        \begin{bmatrix}
            0 & 5 \\
            2 & 5
        \end{bmatrix}
    \begin{bmatrix}
        1 \\
        1
    \end{bmatrix} \\
    = & 12.
\end{align*}
$$

And similarly, $$\boldsymbol{G}_{1, \, 1, \, 2}$$ is given by

$$
\begin{align*}
    & \boldsymbol{e}^\intercal \left( F_{1:\,2, \, 1:\,2, \, 1} \cdot \boldsymbol{K}_{:, \, :, \, 1, \, 2} + F_{1:\,2, \, 1:\,2, 2} \cdot \boldsymbol{K}_{:, \, :, \, 2, \, 2} + F_{1:\,2, \, 1:\,2, 3} \cdot \boldsymbol{K}_{:, \, :, \, 3, \, 2} \right) \boldsymbol{e} \\
    = & \boldsymbol{e}^\intercal \left(
        \begin{bmatrix}
            0 & 3 \\
            0 & 4
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            0 & 1 \\
            0 & 0
        \end{bmatrix}
        +
        \begin{bmatrix}
            0 & 1 \\
            2 & 4
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            0 & 1 \\
            1 & 0
        \end{bmatrix}
        +
        \begin{bmatrix}
            4 & 4 \\
            2 & 1
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            0 & 0 \\
            1 & 0
        \end{bmatrix}
    \right) \boldsymbol{e} \\
    = & \boldsymbol{e}^\intercal \left(
        \begin{bmatrix}
            0 & 0 \\
            0 & 0
        \end{bmatrix}
        +
        \begin{bmatrix}
            1 & 0 \\
            4 & 2
        \end{bmatrix}
        +
        \begin{bmatrix}
            4 & 0 \\
            1 & 0
        \end{bmatrix}
    \right) \boldsymbol{e} \\
    = & \begin{bmatrix}
            1 & 1
        \end{bmatrix}
        \begin{bmatrix}
            5 & 0 \\
            5 & 2
        \end{bmatrix}
    \begin{bmatrix}
        1 \\
        1
    \end{bmatrix} \\
    = & 12.
\end{align*}
$$

As for $$\boldsymbol{G}_{1, \, 2, \, 1}$$, we need to move the first filter $$\boldsymbol{K}_{:, \, :, \, :, \, 1}$$ 1 step right, which means $$\boldsymbol{G}_{1, \, 2, \, 1}$$ is given by

$$
\begin{align*}
    & \boldsymbol{e}^\intercal \left( F_{1:\,2, \, 2:\,3, \, 1} \cdot \boldsymbol{K}_{:, \, :, \, 1, \, 1} + F_{1:\,2, \, 2:\,3, \, 2} \cdot \boldsymbol{K}_{:, \, :, \, 2, \, 1} + F_{1:\,2, \, 2:\,3, \, 3} \cdot \boldsymbol{K}_{:, \, :, \, 3, \, 1} \right) \boldsymbol{e} \\
    = & \boldsymbol{e}^\intercal \left(
        \begin{bmatrix}
            3 & 2 \\
            4 & 1
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            1 & 0 \\
            0 & 0
        \end{bmatrix}
        +
        \begin{bmatrix}
            1 & 4 \\
            4 & 2
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            1 & 0 \\
            0 & 1
        \end{bmatrix}
        +
        \begin{bmatrix}
            4 & 0 \\
            1 & 1
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            0 & 0 \\
            0 & 1
        \end{bmatrix}
    \right) \boldsymbol{e} \\
    = & \boldsymbol{e}^\intercal \left(
        \begin{bmatrix}
            3 & 0 \\
            4 & 0
        \end{bmatrix}
        +
        \begin{bmatrix}
            1 & 4 \\
            4 & 2
        \end{bmatrix}
        +
        \begin{bmatrix}
            0 & 0 \\
            0 & 1
        \end{bmatrix}
    \right) \boldsymbol{e} \\
    = & \begin{bmatrix}
            1 & 1
        \end{bmatrix}
        \begin{bmatrix}
            4 & 4 \\
            8 & 3
        \end{bmatrix}
    \begin{bmatrix}
        1 \\
        1
    \end{bmatrix} \\
    = & 19.
\end{align*}
$$

### Depthwise Separable Convolution

In standard convolution, to tranform $$\boldsymbol{F} \in \mathbb{R}^{D_F \times D_F \times M}$$ into $$\boldsymbol{G} \in \mathbb{R}^{D_G \times D_G \times N}$$, we need $$N$$ separate kernels, each of which has $$M$$ filters. Thus, in total, there are $$N \times M$$ different filters, and each of them requires $$D_K^2 \times D_G^2$$ multiplications. One idea to reduce computation is to let each input channel to share one filter, and the resulted number of filters required can be reduced to $M$. Then, we use pointwise convolution to combine these filtered features into new ones. This idea is called **depthwise separable convolution**, which consists of two operations – a **depthwise convolution** and a **pointwise convolution**.

#### Depthwise Convolution

Let $$\hat{\boldsymbol{K}}$$: $$D_K \times D_K \times M$$ be the depthwise convolutional kernel with one filter per channel and $$\hat{\boldsymbol{G}}$$: $$D_G \times D_G \times M$$ be the corresponding output feature map. Then

$$
\label{eqn4}
\hat{\boldsymbol{G}}_{k, \, l, \, m} = \sum_{i, \, j} \hat{\boldsymbol{K}}_{i, \, j, \, m} \cdot \boldsymbol{F}_{k+i-1, \, l+j-1, \, m},
$$

taking

$$
\label{eqn5}
D_K \cdot D_K \cdot M \cdot D_G \cdot D_G
$$

multiplications.

> **Notice**: Depthwise convolution only filters input channels, so it does not combine them to create new features.

<figure>
  <img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/depthwise_convolution_filters.png" alt="filters of depthwise convolution" style="width:30%">
  <figcaption>Filters of depthwise convolution.</figcaption>
</figure>

Let's use $$\eqref{eqn3}$$ as an example again. Recall that in $$\eqref{eqn3}$$, but now this time, assume

$$
\hat{\boldsymbol{K}}_{:, \, :, \, 1} =
\begin{bmatrix}
    1 & 0 \\
    0 & 0
\end{bmatrix},
\hat{\boldsymbol{K}}_{:, \, :, \, 2} =
\begin{bmatrix}
    1 & 0 \\
    0 & 1
\end{bmatrix},
\hat{\boldsymbol{K}}_{:, \, :, \, 3} =
\begin{bmatrix}
    0 & 0 \\
    0 & 1
\end{bmatrix}. \notag
$$

Notice that there is no summation across the channel in depthwise convolution, so $$\hat{\boldsymbol{G}}_{1, \, 1, \, 1}$$ is simply given by

$$
\boldsymbol{e}^\intercal \cdot \boldsymbol{F}_{1: \, 2, \, 1: \, 2, \, 1} \cdot \hat{\boldsymbol{K}}_{:, \, :, \, 1} \cdot \boldsymbol{e} = 0. \notag
$$

Similarly, $$\hat{\boldsymbol{G}}_{1, \, 2, \, 1}$$ is given by

$$
\boldsymbol{e}^\intercal \cdot \boldsymbol{F}_{1: \, 2, \, 2: \, 3, \, 1} \cdot \hat{\boldsymbol{K}}_{:, \, :, \, 1} \cdot \boldsymbol{e} = 7, \notag
$$

and $$\hat{\boldsymbol{G}}_{1, \, 1, \, 2}$$ is given by

$$
\boldsymbol{e}^\intercal \cdot \boldsymbol{F}_{1: \, 2, \, 1: \, 2, \, 2} \cdot \hat{\boldsymbol{K}}_{:, \, :, \, 2} \cdot \boldsymbol{e} = 7. \notag
$$

#### Pointwise Convolution

Pointwise convolution is a special case of standard convolution, in which the kernel size is set to be $$1 \times 1$$. Let $$\tilde{\boldsymbol{K}}$$: $$1 \times 1 \times M \times N$$ be the **pointwise convolutional kernel** which takes $$\hat{\boldsymbol{G}}$$: $$D_G \times D_G \times M$$ as input and outputs $$\tilde{\boldsymbol{G}}$$: $$D_G \times D_G \times N$$. Then

$$
\label{eqn6}
\tilde{\boldsymbol{G}}_{k, \, l, \, n} = \sum_{m} \tilde{\boldsymbol{K}}_{1, \, 1,\, m, \, n} \cdot \hat{G}_{k, \, l, \, m},
$$

which entails multiplications of

$$
\label{eqn7}
M \cdot N \cdot D_G \cdot D_G
$$

times.

<figure>
  <img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/pointwise_convolution_filters.png" alt="filters of pointwise convolution" style="width:30%">
  <figcaption>Filters of pointwise convolution.</figcaption>
</figure>

Thus, the overall depthwise separable convolution can be expressed as

$$
\label{eqn8}
\tilde{\boldsymbol{G}}_{k, \, l, \, n} = \sum_{m} \tilde{\boldsymbol{K}}_{1, \, 1, \, m, \, n} \cdot \sum_{i, \, j} \hat{\boldsymbol{K}}_{i, \, j, \, m} \cdot \boldsymbol{F}_{k + i - 1, \, l + j - 1, \, m}.
$$

By $$\eqref{eqn5}$$ and $$\eqref{eqn7}$$, the total number of multiplications required in $$\eqref{eqn8}$$ is

$$
\label{eqn9}
D_K \cdot D_K \cdot M \cdot D_G \cdot D_G + M \cdot N \cdot D_G \cdot D_G,
$$

which is $$\frac{1}{N} + \frac{1}{D_K^2}$$ of $$\eqref{eqn2}$$.

### Illustrations


<figure>
  <img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/standard_convolution.png" alt="standard convolution" style="width:40%">
  <figcaption>Illustration of standard convolution.</figcaption>
</figure>

<figure>
  <img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/depthwise_convolution.png" alt="depthwise convolution" style="width:40%">
  <figcaption>Illustration of depthwise convolution.</figcaption>
</figure>

<figure>
  <img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/pointwise_convolution.png" alt="pointwise convolution" style="width:40%">
  <figcaption>Illustration of pointwise convolution.</figcaption>
</figure>

## The Architecture of MobileNetV1

Except that the first layer is a standard convolution, all other convolutions in MobileNetV1 are depthwise separable. Batch normalization and ReLU activation are also used after each convolution.

<figure>
  <img src="/posts.assets/2021-10-08-introduction-to-MobileNetV1.assets/standard_conv_layer_vs_depthwise_separable_conv_layer.png" alt="standard conv layer vs depthwise seperable conv layer" style="width:40%">
  <figcaption>The standard convolutional block and depthwise convolutional blocks used in MobileNetV1.</figcaption>
</figure>

The structure of MobileNetV1 is shown below.

| Type / Stride                   | Filter Shape                 | Input Size                   |
|---------------------------------|------------------------------|------------------------------|
| Conv / s2                       | 3×3×3×32                     | 224×224×3                    |
| Conv dw / s1                    | 3×3×32 dw                    | 112×112×32                   |
| Conv / s1                       | 1×1×32×64                    | 112×112×32                   |
| Conv dw / s2                    | 3×3×64 dw                    | 112×112×64                   |
| Conv / s1                       | 1×1×64×128                   | 56×56×64                     |
| Conv dw / s1                    | 3×3×128 dw                   | 56×56×128                    |
| Conv / s1                       | 1×1×128×128                  | 56×56×128                    |
| Conv dw / s2                    | 3×3×128 dw                   | 56×56×128                    |
| Conv / s1                       | 1×1×128×256                  | 28×28×128                    |
| Conv dw / s1                    | 3×3×256 dw                   | 28×28×256                    |
| Conv / s1                       | 1×1×256×256                  | 28×28×256                    |
| Conv dw / s2                    | 3×3×256 dw                   | 28×28×256                    |
| Conv / s1                       | 1×1×256×512                  | 14×14×256                    |
| 5 × (Conv dw / s1 + Conv / s2)  | (3×3×512 dw) + (1×1×512×512) | (14×14×512) + (14×14×512)    |
| Conv dw / s2                    | 3×3×512 dw                   | 14×14×512                    |
| Conv / s1                       | 1×1×512×1024                 | 7×7×512                      |
| Conv dw / s2                    | 3×3×1024 dw                  | 7×7×1024                     |
| Conv / s1                       | 1×1×1024×1024                | 7×7×1024                     |
| AvgPool / s1                    | 7×7                          | 7×7×1024                     |
| FC / s1                         | 1024×1000                    | 1×1×1024                     |
| Softmax / s1                    | Classifier                   | 1×1×1000                     |

## Experiments

Results from the original paper [*MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*](https://arxiv.org/abs/1704.04861):

| Model                                           | [ImageNet](https://www.image-net.org/) Accuracy | Mult-Adds (M) | Parameters (M) |
| ----------------------------------------------- | ----------------- | ------------- | -------------- |
| [MobileNetV1](https://arxiv.org/abs/1704.04861) | 70.6%             | 569           | 4.2            |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)    | 69.8%             | 1550          | 6.8            |
| [VGG16](https://arxiv.org/abs/1409.1556)        | 71.5%             | 15300         | 138            |

Results on [Keras Applications](https://keras.io/api/applications/):

| Model                                            | Size (MB)         | Top-1 Accuracy | Top-5 Accuracy | Parameters  | Depth | Time (ms) per inference step (CPU) | Time (ms) per inference step (GPU) |
|--------------------------------------------------|-------------------|----------------|----------------|-------------|-------|------------------------------------|------------------------------------|
| [Xception](https://arxiv.org/abs/1610.02357)     | 88                | 0.790          | 0.945          | 22,910,480  | 126   | 109.42                             | 8.06                               |
| [VGG16](https://arxiv.org/abs/1409.1556)         | 528               | 0.713          | 0.901          | 138,357,544 | 23    | 69.50                              | 4.16                               |
| [VGG19](https://arxiv.org/abs/1409.1556)         | 549               | 0.713          | 0.900          | 143,667,240 | 26    | 84.75                              | 4.38                               |
| [ResNet50](https://arxiv.org/abs/1512.03385)     | 98                | 0.749          | 0.921          | 25,636,712  | -     | 58.20                              | 4.55                               |
| [ResNet101](https://arxiv.org/abs/1512.03385)    | 171               | 0.764          | 0.928          | 44,707,176 | -     | 89.59                              | 5.19                               |
| [InceptionV3](https://arxiv.org/abs/1512.00567)  | 92                | 0.779          | 0.937          | 23,851,784  | 159   | 42.25                              | 6.86                               |
| [MobileNetV1](https://arxiv.org/abs/1704.04861)  | 16                | 0.704          | 0.895          | 4,253,864   | 88    | 22.60                              | 3.44                               |

- The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.
- Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.
- Time per inference step is the average of 30 batchs and 10 repetitions.
  - CPU: AMD EPYC Processor (with IBPB) (92 core)
  - Ram: 1.7T
  - GPU: Tesla A100
  - Batch size: 32

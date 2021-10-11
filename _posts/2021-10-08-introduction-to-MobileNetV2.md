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
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 40%;
}
</style>

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

## Inverted Residual Blocks

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

Let $t$ be the **expansion rate**, which indicates by how many times the number of input channels will increase in the expansion layer. Then the structure of a bottleneck with an expansion layer can be summarized in the following table.

| Input                     | Operator               | Output                                       |
|---------------------------|------------------------|----------------------------------------------|
| $h \times w \times k$     | 1×1 conv2d, ReLu6      | $h \times w \times (tk)$                     |
| $h \times w \times (tk)$  | 3×3 dwise s=$s$, ReLU6 | $\frac{h}{s} \times \frac{w}{s} \times (tk)$ |
| $\frac{h}{s} \times \frac{w}{s} \times (tk)$ | 1×1 conv2d | $\frac{h}{s} \times \frac{w}{s} \times k'$ |

### The Structure of Inverted Residual Blocks

The expansion layer in the bottleneck is only utilized to facilitate nonlinear transformation. To preserve information and make backpropagation easier, shortcuts can be added between bottlenecks.

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/inverted_residual_block.png" alt="Inverted Residual Block" class="center4">

- `ReLU6` is exploited as the nonlinearity due to its robustness in low-precision computation ([*MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*](https://arxiv.org/abs/1704.04861)).
- The number of channels in residual blocks ([*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) & [*Aggregated Residual Transformations for Deep Neural Networks*](https://arxiv.org/abs/1611.05431)) drops first and then increases, while the inverted residual block demonstrates a contrary pattern. This is why "inverted" comes from.
- Features output from each convolutional layer are also batch-normalized.

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/bottleneck_residual_block.png" alt="Inverted Residual Block" class="center">

## The Architecture of MobileNetV2

| Input                   | Operator      | $t$ | $c$      | $n$ | $s$ |
|-------------------------|---------------|-----|----------|-----|-----|
| 224×224×3               | conv2d        | -   | 32       | 1   | 2   |
| 112×112×32              | bottleneck    | 1   | 16       | 1   | 1   |
| 112×112×16              | bottleneck    | 6   | 24       | 2   | 2   |
| 56×56×24                | bottleneck    | 6   | 32       | 3   | 2   |
| 28×28×32                | bottleneck    | 6   | 64       | 4   | 2   |
| 14×14×64                | bottleneck    | 6   | 96       | 3   | 1   |
| 14×14×96                | bottleneck    | 6   | 160      | 3   | 2   |
| 7×7×160                 | bottleneck    | 6   | 320      | 1   | 1   |
| 7×7×320                 | conv2d 1×1    | -   | 1280     | 1   | 1   |
| 7×7×1280                | avgpool 7×7   | -   | -        | 1   | -   |
| 1×1×1280                | conv2d 1×1    | -   | k        | -   | -   |

- $t$: the **expansion rate** in the bottleneck;
- $c$: the **number of output channels**;
- $n$: the **number of repetitions**;
- $s$: the **stride**.

There are two types of bottleneck structures, because when $s=2$, the output height and weight will differ from those of the input, so they cannot be concatenated together.

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/two_types_of_bottlenecks.png" alt="Two Types of Bottlenecks" class="center1">

## Experiments

### ImageNet Classification

"CPU" in the table below is the running time in milliseconds for a single core of the Google Pixel 1 phone (using TF-Lite).

| Network                                               | Top 1 Acc | Params   | Mult-Adds | CPU    |
|-------------------------------------------------------|-----------|----------|-----------|--------|
| [MobileNetV1](https://arxiv.org/abs/1704.04861)       | 70.6%     | 4.2M     | 575M      | 113 ms |
| [ShuffleNet](https://arxiv.org/abs/1707.01083) (1.5)  | 71.5%     | **3.4M** | **292M**  | -      |
| [ShuffleNet](https://arxiv.org/abs/1707.01083) (×2)   | 73.7%     | 5.4M     | 524M      | -      |
| [NASNetMobile](https://arxiv.org/abs/1707.07012)      | 74.0%     | 5.3M     | 564M      | 183 ms |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 72.0%     | **3.4M** | 300M      | 75 ms  |
| [MobileNetV2](https://arxiv.org/abs/1801.04381) (1.4) | **74.7%** | 6.9M     | 585M      | 143 ms |

Results on [Keras Applications](https://keras.io/api/applications/):

| Model                                            | Size (MB)         | Top-1 Accuracy | Top-5 Accuracy | Parameters  | Depth | Time (ms) per inference step (CPU) | Time (ms) per inference step (GPU) |
|--------------------------------------------------|-------------------|----------------|----------------|-------------|-------|------------------------------------|------------------------------------|
| [Xception](https://arxiv.org/abs/1610.02357)     | 88                | 0.790          | 0.945          | 22,910,480  | 126   | 109.42                             | 8.06                               |
| [VGG16](https://arxiv.org/abs/1409.1556)         | 528               | 0.713          | 0.901          | 138,357,544 | 23    | 69.50                              | 4.16                               |
| [VGG19](https://arxiv.org/abs/1409.1556)         | 549               | 0.713          | 0.900          | 143,667,240 | 26    | 84.75                              | 4.38                               |
| [ResNet50](https://arxiv.org/abs/1512.03385)     | 98                | 0.749          | 0.921          | 25,636,712  | -     | 58.20                              | 4.55                               |
| [ResNet101](https://arxiv.org/abs/1512.03385)    | 171               | 0.764          | 0.928          | 44,707,176  | -     | 89.59                              | 5.19                               |
| [InceptionV3](https://arxiv.org/abs/1512.00567)  | 92                | 0.779          | 0.937          | 23,851,784  | 159   | 42.25                              | 6.86                               |
| [MobileNetV1](https://arxiv.org/abs/1704.04861)  | 16                | 0.704          | 0.895          | 4,253,864   | 88    | 22.60                              | 3.44                               |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)  | 14                | 0.713          | 0.901          | 3,538,984   | 88    | 25.90                              | 3.83                               |
| [NASNetMobile](https://arxiv.org/abs/1707.07012) | 23                | 0.744          | 0.919          | 5,326,716   | -     | 27.04                              | 6.70                               |

- The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.
- Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.
- Time per inference step is the average of 30 batchs and 10 repetitions.
  - CPU: AMD EPYC Processor (with IBPB) (92 core)
  - Ram: 1.7T
  - GPU: Tesla A100
  - Batch size: 32

### Ablation Study

#### Impact of Nonlinearity in the Bottleneck Layer

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/ablation_study_1.png" alt="Impact of Nonlinearity in the Bottleneck Layer" class="center3">

#### Impact of Shortcuts in Inverted Residual Blocks

<img src="/posts.assets/2021-10-08-introduction-to-MobileNetV2.assets/ablation_study_2.png" alt="Impact of Variations in Residual Blocks" class="center3">

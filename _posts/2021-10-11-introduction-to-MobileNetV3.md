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

MobileNetV3, proposed in [*Searching for MobileNetV3*](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf), is based on the combination of [**hardaware-aware network architecture search** (**NAS**)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf) and [**NetAdapt**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.pdf). These two techniques originate from reinforcement learning, in which both the accuracy and the latency are considered during the design of the reward function. NAS is used to find the global network structures with the starting point of [MnasNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf), while NetAdapt searches the best layer-wise structure by tuning the number of filters of each layer. This third version of MobileNet also introduces a lightweight attention mechanism – [**Squeeze-and-Excitation**](https://arxiv.org/abs/1709.01507), and a new activation function – **h-swish**, into the inverted residual block proposed in [*MobileNetV2: Inverted Residuals and Linear Bottlenecks*](https://ieeexplore.ieee.org/document/8578572). This post will concentrate only on the new residual block and the hard version of **swish**, while how NAS and NetAdapt work will NOT be explained.

## Inverted Residual Blocks with Squeeze-and-Excitation

<img src="/posts.assets/2021-10-11-introduction-to-MobileNetV3.assets/inverted_residual_block_with_se.png" alt="Standard Convolution" class="center6">

## Hard Swish

A nonlinear activation called **swish** was introduced in [*Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning*](https://arxiv.org/abs/1702.03118), [*Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units*](https://openreview.net/forum?id=Bk0MRI5lg) and [*Searching for Activation Functions*](https://arxiv.org/abs/1710.05941) to replace $$\text{ReLU}$$, and the accuracy of the resulted neural network can be significantly improved. Swish is defined as

$$
\label{eqn1}
\text{swish}(x) := x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}.
$$

Since $\eqref{eqn1}$ involves a sigmoid function, computation is much expensive on mobile devices. Thus, authors of MobileNetV3 proposed **h-swish** by approximating sigmoid with its piecewise linear hard analog $$\frac{\text{ReLU6}(x+3)}{6}$$, which means

$$
\label{eqn2}
\text{h-swish}(x) := x \cdot \frac{\text{ReLU6}(x+3)}{6}.
$$

<img src="/posts.assets/2021-10-11-introduction-to-MobileNetV3.assets/sigmoid_vs_hsigmoid.png" alt="Standard Convolution" class="center4">

<img src="/posts.assets/2021-10-11-introduction-to-MobileNetV3.assets/swish_vs_hswish.png" alt="Standard Convolution" class="center4">

### Advantages of Hard Swish

- Optimized implementations of $$\text{ReLU6}$$ are available on virtually all software and hardware frameworks.
- In quantized mode, it eliminates potential numerical precision loss caused by different implementations of the approximate sigmoid.
- In practice, $$\text{h-swish}$$ can be implemented as a piece-wise function to reduce the number of memory accesses driving the latency cost down substantially.

### Other Findings

- The cost of applying nonlinearity decreases as it goes deeper into the network, since each layer activation memory typically halves every time the resolution drops.
- Most of the benefits of $$\text{swish}$$ are realized by using them `only`{:.warning} in the deeper layers. Thus, the authors of MobileNetV3 only use $$\text{h-swish}$$ at the second half of the model.

## Redesigning Expensive Layers

Those current models which are based on MobileNetV2's inverted residual blocks use $1\times1$ convolution as a final layer in order to expand to a higher-dimensional feature space. Although this layer is of crucial importance in enriching features for prediction, it also has a high computational cost. To reduce latency and preserve the high-dimensional features, this layer is moved after the final average pooling. This final set of features is now computed at $1 \times 1$ spatial resolution instead of $7 \times 7$. The resulted design brings nearly free computation and latency.

<img src="/posts.assets/2021-10-11-introduction-to-MobileNetV3.assets/last_stage.png" alt="Standard Convolution" class="center6">

The architecture of MobileNetV2, in which the final set of features is computed at $7 \times 7$ resolution:

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
| `7×7×1280`{:.error}     | avgpool 7×7   | -   | -        | 1   | -   |
| 1×1×1280                | conv2d 1×1    | -   | k        | -   | -   |

- $t$: the **expansion rate** in the bottleneck;
- $c$: the **number of output channels**;
- $n$: the **number of repetitions**;
- $s$: the **stride**.

The architecture of MobileNetV3-Large:

| Input                   | Operator         | Expansion Size | Num. of Output Channels | SE                 | NL | $s$ |
|-------------------------|------------------|----------------|-------------------------|--------------------|----|-----|
| 224×224×3               | conv2d           | -              | 16                      | -                  | HS | 2   |
| 112×112×16              | bneck, 3x3       | 16             | 16                      | :x:                | RE | 1   |
| 112×112×16              | bneck, 3×3       | 64             | 24                      | :x:                | RE | 2   |
| 56x56x24                | bneck, 3x3       | 72             | 24                      | :x:                | RE | 1   |
| 56x56x24                | bneck, 5x5       | 72             | 40                      | :heavy_check_mark: | RE | 2   |
| 28x28x40                | bneck, 5x5       | 120            | 40                      | :heavy_check_mark: | RE | 1   |
| 28x28x40                | bneck, 5x5       | 120            | 40                      | :heavy_check_mark: | RE | 1   |
| 28x28x40                | bneck, 3x3       | 240            | 80                      | :x:                | HS | 2   |
| 14x14x80                | bneck, 3x3       | 200            | 80                      | :x:                | HS | 1   |
| 14x14x80                | bneck, 3x3       | 184            | 80                      | :x:                | HS | 1   |
| 14x14x80                | bneck, 3x3       | 184            | 80                      | :x:                | HS | 1   |
| 14x14x80                | bneck, 3x3       | 480            | 112                     | :heavy_check_mark: | HS | 1   |
| 14x14x112               | bneck, 3x3       | 672            | 112                     | :heavy_check_mark: | HS | 1   |
| 14x14x112               | bneck, 5x5       | 672            | 160                     | :heavy_check_mark: | HS | 2   |
| 7x7x160                 | bneck, 5x5       | 960            | 160                     | :heavy_check_mark: | HS | 1   |
| 7x7x160                 | bneck, 5x5       | 960            | 160                     | :heavy_check_mark: | HS | 1   |
| 7x7x160                 | conv2d, 1x1      | -              | 960                     | -                  | HS | 1   |
| 7x7x960                 | pool, 7x7        | -              | -                       | -                  | -  | 1   |
| `1x1x960`{:.error}      | conv2d, 1x1, NBN | -              | 1280                    | -                  | HS | 1   |
| 1x1x1280                | conv2d, 1x1, NBN | -              | k                       | -                  | -  | 1   |

- SE: whether there is a squeeze-and-excitation in that block;
- NL: the type of activation used;
- HS: $$\text{h-swish}$$;
- RE: $$\text{ReLU}$$;
- NBN: no batch normalization;
- $s$: stride.

Once the cost of this feature generation layer has been mitigated, the previous bottleneck projection layer is no longer needed to reduce computation. This observation allows us to remove the filtering and projection layers in the previous bottleneck layer, further reducing computational complexity. The efficient last stage reduces the latency by 7 ms which is 11% of the running time and the number of operations by 30 M mult-adds with almost no loss of accuracy.

## The Architecture of MobileNetV3

MobileNetV3 has a large version and a small version, which are targeted at high and low resource use cases, respectively.

### MobileNetV3-Large

| Input                   | Operator         | Expansion Size | Num. of Output Channels | SE                 | NL | $s$ |
|-------------------------|------------------|----------------|-------------------------|--------------------|----|-----|
| 224×224×3               | conv2d           | -              | 16                      | -                  | HS | 2   |
| 112×112×16              | bneck, 3x3       | 16             | 16                      | :x:                | RE | 1   |
| 112×112×16              | bneck, 3×3       | 64             | 24                      | :x:                | RE | 2   |
| 56x56x24                | bneck, 3x3       | 72             | 24                      | :x:                | RE | 1   |
| 56x56x24                | bneck, 5x5       | 72             | 40                      | :heavy_check_mark: | RE | 2   |
| 28x28x40                | bneck, 5x5       | 120            | 40                      | :heavy_check_mark: | RE | 1   |
| 28x28x40                | bneck, 5x5       | 120            | 40                      | :heavy_check_mark: | RE | 1   |
| 28x28x40                | bneck, 3x3       | 240            | 80                      | :x:                | HS | 2   |
| 14x14x80                | bneck, 3x3       | 200            | 80                      | :x:                | HS | 1   |
| 14x14x80                | bneck, 3x3       | 184            | 80                      | :x:                | HS | 1   |
| 14x14x80                | bneck, 3x3       | 184            | 80                      | :x:                | HS | 1   |
| 14x14x80                | bneck, 3x3       | 480            | 112                     | :heavy_check_mark: | HS | 1   |
| 14x14x112               | bneck, 3x3       | 672            | 112                     | :heavy_check_mark: | HS | 1   |
| 14x14x112               | bneck, 5x5       | 672            | 160                     | :heavy_check_mark: | HS | 2   |
| 7x7x160                 | bneck, 5x5       | 960            | 160                     | :heavy_check_mark: | HS | 1   |
| 7x7x160                 | bneck, 5x5       | 960            | 160                     | :heavy_check_mark: | HS | 1   |
| 7x7x160                 | conv2d, 1x1      | -              | 960                     | -                  | HS | 1   |
| 7x7x960                 | pool, 7x7        | -              | -                       | -                  | -  | 1   |
| 1x1x960                 | conv2d, 1x1, NBN | -              | 1280                    | -                  | HS | 1   |
| 1x1x1280                | conv2d, 1x1, NBN | -              | k                       | -                  | -  | 1   |

- SE: whether there is a squeeze-and-excitation in that block;
- NL: the type of activation used;
- HS: $$\text{h-swish}$$;
- RE: $$\text{ReLU}$$;
- NBN: no batch normalization;
- $s$: stride.

### MobileNetV3-Small

| Input                   | Operator         | Expansion Size | Num. of Output Channels | SE                 | NL | $s$ |
|-------------------------|------------------|----------------|-------------------------|--------------------|----|-----|
| 224×224×3               | conv2d, 3x3      | -              | 16                      | -                  | HS | 2   |
| 112×112×16              | bneck, 3x3       | 16             | 16                      | :heavy_check_mark: | RE | 2   |
| 56×56×16                | bneck, 3×3       | 72             | 24                      | :x:                | RE | 2   |
| 28x28x24                | bneck, 3x3       | 88             | 24                      | :x:                | RE | 1   |
| 28x28x24                | bneck, 5x5       | 96             | 40                      | :heavy_check_mark: | HS | 2   |
| 14x14x40                | bneck, 5x5       | 240            | 40                      | :heavy_check_mark: | HS | 1   |
| 14x14x40                | bneck, 5x5       | 240            | 40                      | :heavy_check_mark: | HS | 1   |
| 14x14x40                | bneck, 5x5       | 120            | 48                      | :heavy_check_mark: | HS | 1   |
| 14x14x48                | bneck, 5x5       | 144            | 48                      | :heavy_check_mark: | HS | 1   |
| 14x14x48                | bneck, 5x5       | 288            | 96                      | :heavy_check_mark: | HS | 2   |
| 7x7x96                  | bneck, 5x5       | 576            | 96                      | :heavy_check_mark: | HS | 1   |
| 7x7x96                  | bneck, 5x5       | 576            | 96                      | :heavy_check_mark: | HS | 1   |
| 7x7x96                  | conv2d, 1x1      | -              | 576                     | :heavy_check_mark: | HS | 1   |
| 7x7x576                 | pool, 7x7        | -              | -                       | -                  | -  | 1   |
| 1x1x576                 | conv2d, 1x1, NBN | -              | 1280                    | -                  | HS | 1   |
| 1x1x1280                | conv2d, 1x1, NBN | -              | k                       | -                  | -  | 1   |

- SE: whether there is a squeeze-and-excitation in that block;
- NL: the type of activation used;
- HS: $$\text{h-swish}$$;
- RE: $$\text{ReLU}$$;
- NBN: no batch normalization;
- $s$: stride.

## Experiments

### ImageNet Classification

| Network                                      | Top-1     | Mult-Adds (M) | Params (M) | P-1 (ms) | P-2 (ms) | P-3 (ms) |
|----------------------------------------------|-----------|---------------|------------|----------|----------|----------|
| V3-Large 1.0                                 | **75.2%** | 219           | 5.4        | 51       | 61       | 44       |
| V3-Large 0.75                                | 73.3%     | 155           | 4.0        | 39       | 46       | 40       |
| MnasNet-A1                                   | **75.2%** | 315           | 3.9        | 71       | 86       | 61       |
| [Proxyless](https://arxiv.org/abs/1812.00332)| 74.6%     | 320           | 4.0        | 72       | 84       | 60       |
| V2 1.0                                       | 72.0%     | 300           | 3.4        | 64       | 76       | 56       |
| V3-Small 1.0                                 | 67.4%     | 66            | 2.9        | 15.8     | 19.4     | 14.4     |
| V3-Small 0.75                                | 65.4%     | 44            | 2.4        | 12.8     | 15.6     | 11.7     |
| Mnas-small                                   | 64.9%     | 65.1          | 1.9        | 20.3     | 24.2     | 17.2     |
| V2 0.35                                      | 60.8%     | 59.2          | 1.6        | 16.6     | 19.6     | 13.9     |

- P-$n$: a Pixel-$n$ phone.
- All latencies are measures using a single large core with a batch size of 1.

The trade-off between the number of mult-adds and top-1 accuracy is shown below. This allows to compare models that were targeted different hardware or software frameworks. All MobileNetV3s are for input resolution 224 and use multipliers 0.35, 0.5, 0.75, 1 and 1.25.

<img src="/posts.assets/2021-10-11-introduction-to-MobileNetV3.assets/model_comparison.png" alt="Standard Convolution" class="center4">

#### MobileNetV2 vs. MobileNetV3

| Network      | Top-1 | P-1  | P-2  | P-3  |
|--------------|-------|------|------|------|
| V3-Large 1.0 | 73.8% | 44   | 42.5 | 31.7 |
| V2 1.0       | 70.9% | 52   | 48.3 | 37.0 |
| V3-Small     | 64.9% | 15.5 | 14.9 | 10.7 |
| V2 0.35      | 57.2% | 16.7 | 15.6 | 11.9 |

The figure below shows the MobileNetV3 performance trade-offs as a function of multiplier and resolution. Note how MobileNetV3-Small outperforms the MobileNetV3- Large with multiplier scaled to match the performance by nearly 3%. On the other hand, resolution provides an even better trade-offs than multiplier. However, it should be noted that resolution is often determined by the problem (e.g. segmentation and detection problem generally require higher resolution), and thus can’t always be used as a tunable parameter. In this experiment, multipliers are set to be 0.35, 0.5, 0.75, 1.0 and 1.25, with a fixed resolution of 224, and resolutions 96, 128, 160, 192, 224 and 256 with a fixed depth multiplier of 1.0.

<img src="/posts.assets/2021-10-11-introduction-to-MobileNetV3.assets/v2_vs_v3.png" alt="Standard Convolution" class="center3">

### Ablation Study

#### Impact of Nonlinearities

Effects of nonlinearities on MobileNetV3-Large is shown in the table below.

| Model         | Top-1                      | P-1 (ms)                | P-1 (no-opt; ms) |
|---------------|----------------------------|-------------------------|------------------|
| V3-Large 1.0  | 75.2%                      | 51.4                    | 57.5             |
| ReLU          | 74.5% (`-0.7%`{:.error})   | 50.5 (`-1%`{:.success}) | 50.5             |
| h-swish @16   | 75.4% (`+0.2%`{:.success}) | 53.4 (`+4%`{:.error})   | 68.9             |
| h-swish @112  | 75.0% (`-0.3%`{:.error})   | 51.0 (`-0.5%`{:.success})| 54.4             |

- In $$\text{h-swish} \, @ N$$, $N$ denotes the number of channels in the first layer that has $$\text{h-swish}$$ enabled.
- Top-1 accuracy is measured on ImageNet.
- The third column shows the runtime without optimized $$\text{h-swish}$$.

#### Impact of Other Components

<img src="/posts.assets/2021-10-11-introduction-to-MobileNetV3.assets/progression.png" alt="Standard Convolution" class="center4">

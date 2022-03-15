---
key: 2022_03_15_01
title: Introduction to Variational Autoencoders
tags: ["Computer Vision", "Self-Supervised Learning"]
mathjax: true
mathjax_autoNumber: true
author: Yiming
comment: false
pageview: false
aside:
    toc: true
---

> This post refers to many materials of [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) by Joseph Rocca and Baptiste Rocca. The original paper in which VAEs are proposed is called *Auto-Encoding Variational Bayes* by Diederik Kingma et al.

# Introduction

In the last few years, deep-learning-based generative models have gained more and more interest due to some amazing improvements in the field. Relying on

- a huge amount of data,
- well-designed network architectures,
- smart training techniques,

deep generative models have shown an incredible ability to produce highly realistic pieces of content of various kind, such as images, texts and sounds. Among these deep generative models, two major families stand out and deserve a special attention: 

- **Generative Adversarial Networks** (**GANs**),
- **Variational Autoencoders** (**VAEs**).

<figure>
  <img src="/posts.assets/2022-03-15-variational-autoencoders/visualisations.png" alt="visualisations" style="width:100%">
  <figcaption>Face images generated from a variational autoencoder.</figcaption>
</figure>

In  [Introduction to Generative Adversarial Networks](https://yimingma.github.io/2022/03/08/generative-adversarial-nets.html), we showed in GANs how adversarial training can oppose the two networks – a generator and a discriminator, to push both of them to be improved after each iteration.
Now we introduce the other major kind of deep generative models: variational autoencoders. In a nutshell, a VAE is an autoencoder whose encoding distribution is regularised during the training in order to ensure that its latent space has good properties in generating new data. Moreover, the term “variational” comes from the close relation between such regularisation and the variational inference method in statistics.
If the last two sentences summarise pretty well the notion of VAEs, they can also raise a lot of questions.

- What is an autoencoder?
- What is the latent space and why regularising it?
- How to generate new data from VAEs?
- What is the link between VAEs and variational inference?

To describe VAEs as well as possible, we will try to answer all this questions (and many others!) and to provide the reader with as many insights as we can (ranging from basic intuitions to more advanced mathematical details). Without further ado, let’s (re)discover VAEs together!

# Dimensionality Reduction

In this section we will discuss some concepts related to **dimensionality reduction**. In particular, we will briefly review **principal component analysis** (**PCA**) and **autoencoders**, showing how both ideas are related to each others.

## What Is Dimensionality Reduction?

<span style="color:RoyalBlue;">In machine learning, [**dimensionality reduction**](https://en.wikipedia.org/wiki/Dimensionality_reduction) is the process of reducing the number of features that describe some data. This reduction is done either by</span>

- <span style="color:RoyalBlue;">**selection** (only some existing features are conserved)</span>

<span style="color:RoyalBlue;">or by</span>

- <span style="color:RoyalBlue;">**extraction** (a reduced number of new features are created based on the old ones).</span>

Dimensionality reduction can be useful in many situations that require low-dimensional data (e.g., data visualisation, data storage, heavy computation). Although there exist many different methods to reduce dimensionality, we can set a global framework that is matched by most of these methods.

First, let’s call <span style="color:RoyalBlue;">**encoder** the process that produce the new features from the “old-feature” representation (by selection or by extraction)</span> and <span style="color:RoyalBlue;">**decoder** the reverse process</span>. 
<span style="color:Coral;">Dimensionality reduction can then be interpreted as *data* *compression* where the encoder compress the data (from the initial space to the **encoded space**, also called **latent space**) whereas the decoder decompress them. Of course, depending on</span>

- <span style="color:Coral;">the initial data distribution</span>
- <span style="color:Coral;">the latent space dimension</span>
- <span style="color:Coral;">the encoder definition</span>

<span style="color:Coral;">this compression can be *lossy*, meaning that some information is lost during encoding and cannot be recovered after decoding.</span>

<figure>
  <img src="/posts.assets/2022-03-15-variational-autoencoders/dimensionality_reduction.png" alt="dimensionality reduction" style="width:100%">
  <figcaption>Illustration of the dimensionality reduction principle with the encoder and the decoder.</figcaption>
</figure>

<span style="color:Crimson;">The main purpose of a dimensionality reduction method is to find the best encoder & decoder pair among a given family. In other words, for a given set of possible encoders and decoders, we are looking for the pair that *keeps the maximum of information when encoding (so the reconstruction error of decoding is also minimum).*</span> If we denote respectively $\mathcal{E}$ and $\mathcal{D}$ the families of encoders and decoders we are considering, then the dimensionality reduction problem can be written

$$
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
(\hat{\boldsymbol{E}}, \, \hat{\boldsymbol{D}}) = \argmin_{(\boldsymbol{E}, \,\boldsymbol{D}) \in \boldsymbol{\mathcal{E}} \times \boldsymbol{\mathcal{D}}} \epsilon \left(\boldsymbol{X}, \, \boldsymbol{D}\left(\boldsymbol{E}(\boldsymbol{X})\right)\right), \notag
$$

where

$$
\epsilon \left(\boldsymbol{X}, \, \boldsymbol{D}\left(\boldsymbol{E}(\boldsymbol{X})\right)\right) \notag
$$

defines the reconstruction error measured between the input data $\boldsymbol{X}$ and the encoded-decoded data $\boldsymbol{D} ( \boldsymbol{E} ( \boldsymbol{X} ))$.

In the following sections, we will denote

- $N$: the size of data;
- $n_d$: the dimension of the initial (decoded) space;
- $n_e$: the dimension of the reduced (encoded) space.

## Autoencoders

Let’s now discuss autoencoders and see how we can use neural networks for dimensionality reduction. <span style="color:Crimson;">The general idea of autoencoders is pretty simple and consists in setting an encoder and a decoder as *neural networks* whose parameters can be learned iteratively.</span> In each iteration, we first feed the autoencoder (the encoder followed by the decoder) with some data. Then we compare the encoded-decoded output with the initial data and backpropagate the error to update parameters.
<span style="color:Coral;">Thus, intuitively, the overall autoencoder architecture creates a *bottleneck* for data that ensures only the main structured part of the information can go through and be reconstructed.</span> Looking at our general framework, 

- the family $\mathcal{E}$ of considered encoders is defined by the encoder network architecture;
- the family $\mathcal{D}$ of considered decoders is defined by the decoder network architecture;
- the search of encoder and decoder that minimise the reconstruction error is done by gradient descent over the parameters of these networks.

<figure>
  <img src="/posts.assets/2022-03-15-variational-autoencoders/autoencoder_structure.png" alt="autoencoder structure" style="width:100%">
  <figcaption>Illustration of an autoencoder with its loss function.</figcaption>
</figure>
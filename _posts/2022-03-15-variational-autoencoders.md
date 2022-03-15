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

## Introduction

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

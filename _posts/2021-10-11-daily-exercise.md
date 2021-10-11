---
key: 2021_10_11_02
title: Computing Expectations by Conditioning
tags: ["Probability", "Daily Exercise"]
mathjax: true
mathjax_autoNumber: true
author: Yiming
comment: false
pageview: false
aside:
    toc: true
---

## $$\mathbb{E} [X] = \mathbb{E} \left[ \mathbb{E} [X | Y] \right]$$

**Theorem**: let $X$ and $Y$ be two random variables; then $$\mathbb{E} [X | Y]$$ is a function of $Y$, whose value at $Y = y$ is $$\mathbb{E} [X | Y = y]$$. Note that $$\mathbb{E} [X | Y]$$ itself is also a random variable, and the following equation holds:

$$
\mathbb{E} [X] = \mathbb{E} \left[ \mathbb{E} [X | Y] \right]. \notag
$$

**Exercise**: Independent trials, each of which is a success with probability $p$, are performed until there are $k$ consecutive successes. What is the mean number of necessary trials? (From p.116 of *Introduction to Probability Models* by Sheldon M. Ross, published by Elsevier in 2019.)
**Solution**: Let $N_k$ denote the number of necessary trials to obtain $k$ consecutive successes, and let $$M_k := \mathbb{E}[N_k]$$. We will determine $M_k$ by deriving and then solving a recursive equation that it satisfies. To begin, write

$$
\label{eqn1}
N_k = N_{k-1} + A_{k-1, \, k},
$$

where $N_{k-1}$ is the number of trials needed for $k-1$ consecutive successes, and $A_{k-1, \, k}$ is the number of additional trials needed to go from having $k-1$ successes in a rwo to having $k$ in a row. Taking expectations gives that,

$$
\label{eqn2}
M_k = M_{k-1} + \mathbb{E} [A_{k-1, \, k}].
$$

Let $X_{N_{k-1} + 1}$ be the trial after there have been $k-1$ successes in a row. If it is a success ($X_{N_{k-1} + 1}=1$), then that gives $k$ in a row and no additional trials after that are needed; if it is a failure ($X_{N_{k-1} + 1}=0$), then at that point we are starting all over again, so the expected additional number from then on would be $$\mathbb{E}[N_k]$$. Thus,

$$
\begin{align}
\mathbb{E}[A_{k-1, \, k}] = & \mathbb{E} \left[ \mathbb{E}[A_{k-1, \, k} | X_{N_{k-1} + 1}]  \right] \notag \\
= & \mathbb{P} \left\{ X_{N_{k-1} + 1} = 1 \right\} \cdot \mathbb{E}[A_{k-1, \, k} | X_{N_{k-1} + 1} = 1] + \mathbb{P} \left\{ X_{N_{k-1} + 1} = 0 \right\} \cdot \mathbb{E}[A_{k-1, \, k} | X_{N_{k-1} + 1} = 0] \notag \\
= & \frac{1}{2} \cdot 1 + \frac{1}{2} \cdot \left( 1 + \mathbb{E}[M_k] \right) \notag \\
= & 1 + (1-p) M_k \label{eqn3}
\end{align}
$$

Using $\eqref{eqn3}$ in $\eqref{eqn2}$ gives

$$
M_k = M_{k-1} + 1 + (1 - p) M_k \notag,
$$

or

$$
\label{eqn4}
M_k = \frac{1}{p} + \frac{M_{k-1}}{p}.
$$

Since $N_1$, which is the time of the first success, follows a geometric distribution with parameter $p$, we see that

$$
M_1 = \frac{1}{p} \notag
$$

and, recursively

$$
\begin{align*}
M_2 = & \frac{1}{p} + \frac{1}{p^2}, \\
M_3 = & \frac{1}{p} + \frac{1}{p^2} + \frac{1}{p^3}
\end{align*}
$$

and, in general,

$$
M_k = \sum_{i=1}^k \frac{1}{p^i}.
$$

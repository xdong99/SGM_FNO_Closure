
# Data-Driven Stochastic Closure Modeling via Conditional Diffusion Model and Neural Operator

This repo contains the official implementation for the paper [Data-Driven Stochastic Closure Modeling via Conditional Diffusion Model and Neural Operator](https://arxiv.org/abs/2408.02965#:~:text=Data%2DDriven%20Stochastic%20Closure%20Modeling%20via%20Conditional%20Diffusion%20Model%20and%20Neural%20Operator,-Xinghao%20Dong%2C%20Chuanqi&text=Closure%20models%20are%20widely%20used,scales%20is%20often%20too%20expensive.)

by [Xinghao Dong](https://xdong99.github.io/), [Chuanqi Chen](https://github.com/ChuanqiChenCC), and [Jin-Long Wu](https://www.jinlongwu.org/).

--------------------

In this work, we present a data-driven modeling framework to
build stochastic and non-local closure models based on the conditional diffusion model and
neural operator. More specifically, the Fourier neural operator is used to approximate the
score function for a score-based generative diffusion model, which captures the conditional
probability distribution of the unknown closure term given some dependent information,
e.g., the numerically resolved scales of the true system, sparse experimental measurements
of the true closure term, and estimation of the closure term from existing physics-based
models. Fast sampling algorithms are also investigated to ensure the efficiency of the proposed framework. 

![schematic](Assets/Schematic.jpg)

A comprehensive study is performed on the 2-D Navier–Stokes equation, for which the stochastic viscous diffusion 
term is assumed to be unknown. The proposed methodology provides a systematic approach via generative machine learning 
techniques to construct data-driven stochastic closure models for multiscale dynamical systems with 
continuous spatiotemporal fields.

## Citations
```
@article{dong2024data,
  title={Data-Driven Stochastic Closure Modeling via Conditional Diffusion Model and Neural Operator},
  author={Dong, Xinghao and Chen, Chuanqi and Wu, Jin-Long},
  journal={arXiv preprint arXiv:2408.02965},
  year={2024}
}
```

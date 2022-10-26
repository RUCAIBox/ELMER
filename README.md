# ELMER
This repository contains code and checkpoints for **ELMER**:

[**ELMER: A Non-Autoregressive Pre-trained Language Model for Efficient and Effective Text Generation**](https://arxiv.org/abs/2210.13304)

Junyi Li, Tianyi Tang, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen

# Introduction

To explicitly learn the bi-directional token dependency, we propose ELMER: an Efficient and Effective PLM for NAR text generation, which generates tokens at different layers by leveraging the early exit technique.

<div align=center><img src="asset/model.png" alt="Cover" width="50%"/></div>

The architecture of ELMER is a variant of the standard Transformer encoder-decoder and poses three technical contributions:

1. For decoder, we replace the original masked multi-head attention with bi-directional multi-head attention akin to the encoder. Therefore, ELMER dynamically adjusts the output length by emitting an end token `[EOS]` at any position.

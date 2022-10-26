# ELMER
This repository contains code and checkpoints for **ELMER**:

[**ELMER: A Non-Autoregressive Pre-trained Language Model for Efficient and Effective Text Generation**](https://arxiv.org/abs/2210.13304)

Junyi Li, Tianyi Tang, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen

## Introduction

To explicitly learn the bi-directional token dependency, we propose ELMER: an Efficient and Effective PLM for NAR text generation, which generates tokens at different layers by leveraging the early exit technique.

<div align=center><img src="asset/model.png" alt="Cover" width="60%"/></div>

The architecture of ELMER is a variant of the standard Transformer encoder-decoder and poses three technical contributions:

1. For decoder, we replace the original masked multi-head attention with bi-directional multi-head attention akin to the encoder. Therefore, ELMER dynamically adjusts the output length by emitting an end token "[EOS]" at any position.
2. Leveraging early exit, ELMER injects "off-ramps" at each decoder layer, which make predictions with intermediate hidden states. If ELMER exits at the $l$-th layer, we copy the $l$-th hidden states to the subsequent layers.
3. ELMER utilizes a novel pre-training objective, layer permutation language modeling (LPLM), to pre-train on the large-scale corpus. LPLM permutes
the exit layer for each token from 1 to the maximum layer $L$.

## Pre-trained Models

We provide the checkpoint for ELMER-base, which was pre-trained on 16GB English corpus, i.e., BookCorpus and Wikipedia.

- [ELMER-base](): 6 layers encoder, 6 layers decoder, 12 attention heads, and 768 hidden dimensions.

The checkpoint can be directly used with Hugging Face Transformers. In the future, we will integrate ELMER into [Hugging Face](https://huggingface.co/) and [TextBox](https://github.com/RUCAIBox/TextBox) libraries for easy-to-use.

## Requirements

To install requirements

```shell
bash install.sh
```

## How to use

The pre-training code can be found [here](pre-train), and the fine-tuning code can be found [here](fine-tune).

To pre-train or fine-tune ELMER, please copy the file `modeling_bart.py` from the `pre-train` or `fine-tune` directory to the `BART` directory in Transformers, such as `~/miniconda3/envs/[env_name]/lib/python3.7/site-packages/transformers/models/bart`.

```python
from transformers import BartTokenizer as ElmerTokenizer
from transformers import BartForConditionalGeneration as ElmerForConditionalGeneration

# pretrained_model/elmer-base is the saved directory for ELMER checkpoints
tokenizer = ElmerTokenizer.from_pretrained("pretrained_model/elmer-base")
model = ElmerForConditionalGeneration.from_pretrained("pretrained_model/elmer-base")

#--------------------------------
# do training for many many steps
#--------------------------------
```

For example, we'd like to fine-tune ELMER on XSUM dataset:

```python
python train.py --dataset=XSUM --model=ELMER-XSUM --data-dir=[DATASET_DIR] \
       --pretrained_model_dir=[ELMER_BASE_DIR] --saved_dir=[FINE_TUNED_MODEL_DIR] --log-dir=[LOG_DIR] \
       --start_epoch=0 --epochs=100 --train_batch_size=32 --eval_batch_size=32 --optimizer=adam --lr=2e-5
```

These hyper-parameters can be also defined in `config.yaml` file.

## Evaluation

To evaluate the generated texts, the `BLEU`, `METEOR`, and `Distinct` metrics can be computed using our provided scripts in `pyeval` directory. For the `ROUGE` metric, please install the [files2rouge](https://github.com/pltrdy/files2rouge) package and compute it.

## Contact

If you have any problems, raise an issue or contact <lijunyi@ruc.edu.cn>.

## Citation

```bibtex
@article{lijunyi2022elmer,
  title={ELMER: A Non-Autoregressive Pre-trained Language Model for Efficient and Effective Text Generation},
  author={Li, Junyi and Tang, Tianyi and Zhao, Wayne Xin and Nie, Jian-Yun and Wen, Ji-Rong},
  booktitle={EMNLP 2022},
  year={2022}
}
```

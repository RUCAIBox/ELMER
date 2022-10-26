# Pre-training ELMER

In this directory, we will present how to pre-train ELMER on the 16G English corpus, BookCorpus and Wikipedia.

## Data Processing

You should first pre-process the pre-training corpus using two text corruption methods, i.e., sentence shuffling and text infilling.

```python
python corpus_corruption.py
```

We will process and generate 10 different pre-training text copies `{epoch}.json` for each epoch:

- `data/books_wiki/books_wiki.txt`: This file is the original texts for BookCorpus and Wikipedia. The format is one text per line.
- `pretrained_model/bart-base`: This is the BART-base directory from Transformers. ELMER adopts the tokenizer and vocabulary from BART-base.

The pre-processing step is conducted using multiple processes to accelerate the speed.

## Training

After preparing the pre-training corpus, you can set the hyper-parameters in `config.yaml` and start training:

```python
python train.py
```

*Note: you should copy the file `modeling_bart.py` to the BART directory in Transformers.*

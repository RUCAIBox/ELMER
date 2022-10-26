# Pre-training ELMER

In this directory, we will present how to pre-train ELMER on the 16G English corpus, BookCorpus and Wikipedia.

## Data Processing

You should first pre-process the pre-training corpus using two text corruption methods, i.e., sentence shuffling and text infilling.

```python
python corpus_corruption.py
```

We will process and generate `10` different pre-training text copies for each epoch:

- `books_wiki.txt`: This file is the original texts for BookCorpus and Wikipedia. The format is one text per line.
- `bart-base`: This is the BART-base directory from Transformers. ELMER adopts the tokenizer and vocabulary from BART-base.


# Fine-tuning ELMER

In this directory, we will present how to fine-tune ELMER on downstream tasks and datasets.

## Data Processing

You should first pre-process the downstream datasets into `train.src`, `train.tgt`, `valid.src`, `valid.tgt`, `test.src`, and `test.tgt`. The format of these files is one text per line.


## Training

After preparing the pre-training corpus, you can set the hyper-parameters in `config.yaml` and start training:

```python
python train.py
```

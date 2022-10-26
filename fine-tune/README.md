
# Fine-tuning ELMER

In this directory, we will present how to fine-tune ELMER on downstream tasks and datasets.

## Data Processing

You should first pre-process the downstream datasets into `train.src`, `train.tgt`, `valid.src`, `valid.tgt`, `test.src`, and `test.tgt`. The format of these files is one text per line.


## Training

After preparing the downstream datasets, you can set the hyper-parameters in `config.yaml` and start training:

```python
python train.py
```

*Note: you should copy the file `modeling_bart.py` to the BART directory in Transformers.*

## Evaluation

To evaluate the generated texts, the `BLEU`, `METEOR`, and `Distinct` metrics can be computed using our provided scripts in pyeval directory. For the ROUGE metric, please install the [files2rouge](https://github.com/pltrdy/files2rouge) package and compute it. The file `install.sh` has contained the installation step for files2rouge.



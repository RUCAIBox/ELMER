import os
import torch
import time
import numpy as np
from logging import getLogger
from data import S2SDataset
from utils import build_optimizer, init_seed, init_logger, read_configuration, collate_fn_seq2seq, format_time, init_device
from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from eval import Evaluate
from data import BOS_ID, EOS_ID, PAD_ID, MASK_ID


def train(config):
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    logger.info("Initialize ELMER from {}.".format(config["pretrained_model_dir"]))

    tokenizer = BartTokenizer.from_pretrained(config["pretrained_model_dir"])
    model = BartForConditionalGeneration.from_pretrained(config["pretrained_model_dir"]).cuda()
    optimizer = build_optimizer(config, model)

    logger.info("Create training dataset from {}.".format(config["data_dir"]))
    train_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], tokenizer=tokenizer, use_retrieval=config["retrieval"], mode="train"),
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn_seq2seq,
        pin_memory=True)

    logger.info("Create validation dataset from {}.".format(config["data_dir"]))
    valid_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], tokenizer=tokenizer, use_retrieval=config["retrieval"], mode="valid"),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn_seq2seq,
        pin_memory=True)

    best_valid_loss = None
    best_valid_bleu = None
    best_valid_rouge = None
    for epoch_idx in range(config["start_epoch"], config["epochs"]):
        model.train()
        train_loss = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            source_input_ids, source_mask, target_input_ids, target_mask, labels = batch

            source_input_ids = source_input_ids.cuda()
            source_mask = source_mask.cuda()
            target_input_ids = target_input_ids.cuda()
            target_mask = target_mask.cuda()
            labels = labels.cuda()
            output_dict, exited_layers = model(input_ids=source_input_ids,
                                               attention_mask=source_mask,
                                               decoder_input_ids=target_input_ids,
                                               decoder_attention_mask=target_mask,
                                               labels=labels,
                                               return_dict=True)

            loss = output_dict["loss"]
            avg_layer = exited_layers.float().mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.info("Epoch {} batch {} time {}: loss {:.4f}, average layer {:.4f}".format(epoch_idx, batch_idx, format_time(time.time() - t0), loss.item(), avg_layer))

            train_loss += loss.item()
            t0 = time.time()

        train_loss /= len(train_dataloader)
        train_ppl = np.exp(train_loss)
        training_time = format_time(time.time() - t0)
        logger.info("\n\nEpoch {}: training generation loss {} perplexity {} time {}.\n".format(epoch_idx,
                                                                                                train_loss,
                                                                                                train_ppl,
                                                                                                training_time))

        with torch.no_grad():
            model.eval()
            valid_loss = 0
            generated_text = []
            reference_text = []
            calculator = Evaluate()
            t0 = time.time()
            for batch in valid_dataloader:
                source_input_ids, source_mask, target_input_ids, target_mask, labels = batch

                source_input_ids = source_input_ids.cuda()
                source_mask = source_mask.cuda()
                target_input_ids = target_input_ids.cuda()
                target_mask = target_mask.cuda()
                labels = labels.cuda()
                output_dict, exited_layers = model(input_ids=source_input_ids,
                                                   attention_mask=source_mask,
                                                   decoder_input_ids=target_input_ids,
                                                   decoder_attention_mask=target_mask,
                                                   labels=labels,
                                                   return_dict=True)

                loss = output_dict["loss"]
                valid_loss += loss.item()

                logits = output_dict["logits"]
                generated_ids = logits.argmax(dim=-1)
                generated_list = tokenizer.batch_decode(generated_ids)
                generated = []
                for sentence in generated_list:
                    try:
                        end = sentence.index("</s>")
                        generated.append(sentence[:end])
                    except ValueError:
                        generated.append(sentence)

                reference = tokenizer.batch_decode(labels, skip_special_tokens=True)
                generated_text.extend(generated)
                reference_text.extend(reference)
            valid_time = format_time(time.time() - t0)

            valid_loss /= len(valid_dataloader)
            assert len(generated_text) == len(reference_text)
            generated_text = [text.lower().strip() for text in generated_text]
            reference_text = [text.lower().strip() for text in reference_text]
            metric_dict = calculator.evaluate(generated_text, reference_text)
            logger.info("\n\nEpoch {} time {}: validation loss {} "
                        "BLEU-1 {} BLEU-2 {} BLEU-4 {} ROUGE-L {}.\n".format(epoch_idx, valid_time, valid_loss,
                                                                   metric_dict["Bleu_1"], metric_dict["Bleu_2"],
                                                                   metric_dict["Bleu_4"], metric_dict["ROUGE_L"]))

        save = False
        if best_valid_loss is None or best_valid_loss[1] > valid_loss:
            best_valid_loss = (epoch_idx, valid_loss)
            save = True

        if best_valid_bleu is None or best_valid_bleu[1] < metric_dict["Bleu_1"]:
            best_valid_bleu = (epoch_idx, metric_dict["Bleu_1"])
            save = True

        if best_valid_rouge is None or best_valid_rouge[1] < metric_dict["ROUGE_L"]:
            best_valid_rouge = (epoch_idx, metric_dict["ROUGE_L"])
            save = True

        if save:
            saved_path = os.path.join(config["saved_dir"], config["model"], str(epoch_idx))
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            # save pretrained language model
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(saved_path)
            tokenizer.save_pretrained(saved_path)
            logger.info("Save NAR PLM model into {}.".format(saved_path))

    logger.info("\n\nThe best loss {} in epoch {}, the best BLEU-1 {} in epoch {}, "
                "the best ROUGE_L {} in epoch {}.\n".format(best_valid_loss[1], best_valid_loss[0],
                                                          best_valid_bleu[1], best_valid_bleu[0],
                                                          best_valid_rouge[1],best_valid_rouge[0]))


def test(config):
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])

    logger.info("Load fine-tuned NAR model from {}.".format(config["finetuned_model_dir"]))
    tokenizer = BartTokenizer.from_pretrained(config["finetuned_model_dir"])
    model = BartForConditionalGeneration.from_pretrained(config["finetuned_model_dir"]).cuda()
    model.train(False)

    logger.info("Create testing dataset from {}.".format(config["data_dir"]))
    test_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], tokenizer=tokenizer, use_retrieval=config["retrieval"], mode="test"),
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn_seq2seq,
        pin_memory=True)

    calculator = Evaluate()
    model.eval()
    generated_text = []
    reference_text = []
    with torch.no_grad():
        for batch in test_dataloader:
            source_input_ids, source_mask, target_input_ids, target_mask, labels = batch

            source_input_ids = source_input_ids.cuda()
            source_mask = source_mask.cuda()
            target_input_ids = target_input_ids.cuda()
            target_mask = target_mask.cuda()
            labels = labels.cuda()
            output_dict, exited_layers = model(input_ids=source_input_ids,
                                                     attention_mask=source_mask,
                                                     decoder_input_ids=target_input_ids,
                                                     decoder_attention_mask=target_mask,
                                                     labels=labels,
                                                     return_dict=True)
            final_logits = output_dict["logits"]
            generated_ids = final_logits.argmax(dim=-1)
            generated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference = tokenizer.batch_decode(labels, skip_special_tokens=True)
            generated_text.extend(generated)
            reference_text.extend(reference)

    assert len(generated_text) == len(reference_text)
    saved_file_path = os.path.join(config["output_dir"], "output.res")
    fout = open(saved_file_path, "w")
    for i in range(len(generated_text)):
        fout.write("Generated text: " + generated_text[i].strip() + "\n")
        fout.write("Reference text: " + reference_text[i].strip() + "\n")
    fout.close()

    metric_dict = calculator.evaluate(generated_text, reference_text)
    logger.info("\n\nTest evaluation: BLEU-1/2/3/4 {}/{}/{}/{}, ROUGE-L {}.\n".format(metric_dict["Bleu_1"],
                                                                                      metric_dict["Bleu_2"],
                                                                                      metric_dict["Bleu_3"],
                                                                                      metric_dict["Bleu_4"],
                                                                                      metric_dict["ROUGE_L"]))


def main():
    config = read_configuration("config.yaml")

    if config["train"]:
        train(config)
    else:
        test(config)


if __name__ == '__main__':
    main()


import os
import torch
import time
import json
import numpy as np
from logging import getLogger
from data import S2SDataset
from utils import build_optimizer, init_seed, init_logger, read_configuration, collate_fn_seq2seq, format_time, init_device
from transformers import BartTokenizer as ElmerTokenizer
from transformers import BartForConditionalGeneration as ElmerForConditionalGeneration
from torch.utils.data import DataLoader
from eval import Evaluate
from data import BOS_ID, EOS_ID, PAD_ID, MASK_ID


def train(config):
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    logger.info("Initialize ELMER model from {}.".format(config["pretrained_model_dir"]))

    tokenizer = ElmerTokenizer.from_pretrained(config["pretrained_model_dir"])
    model = ElmerForConditionalGeneration.from_pretrained(config["pretrained_model_dir"]).to(device)
    optimizer = build_optimizer(config, model)

    logger.info("Create training dataset from {}.".format(config["data_dir"]))
    train_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], tokenizer=tokenizer, data_usage="train"),
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn_seq2seq,
        pin_memory=True)

    logger.info("Create validation dataset from {}.".format(config["data_dir"]))
    valid_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], tokenizer=tokenizer, data_usage="valid"),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn_seq2seq,
        pin_memory=True)

    best_valid_loss = None
    best_valid_bleu = None
    best_valid_rouge = None
    best_valid_meteor = None
    for epoch_idx in range(config["start_epoch"], config["epochs"]):
        model.train()
        train_loss = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            source_input_ids, source_mask, target_input_ids, target_mask, labels = batch

            source_input_ids = source_input_ids.to(device)
            source_mask = source_mask.to(device)
            target_input_ids = target_input_ids.to(device)
            target_mask = target_mask.to(device)
            labels = labels.to(device)
            output_dict, inter_losses = model(input_ids=source_input_ids,
                                               attention_mask=source_mask,
                                               decoder_input_ids=target_input_ids,
                                               decoder_attention_mask=target_mask,
                                               labels=labels,
                                               return_dict=True)

            loss = 0.0
            loss += output_dict["loss"]
            for inter_loss in inter_losses:
                loss += inter_loss
            loss /= (len(inter_losses) + 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.info("Epoch {} batch {} time {}: loss {:.4f}".format(epoch_idx, batch_idx, format_time(time.time() - t0), loss.item()))

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

                source_input_ids = source_input_ids.to(device)
                source_mask = source_mask.to(device)
                target_input_ids = target_input_ids.to(device)
                target_mask = target_mask.to(device)
                labels = labels.to(device)
                output_dict, inter_losses = model(input_ids=source_input_ids,
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
                        gen = sentence[:end]
                    except ValueError:
                        gen = sentence
                    generated.append(gen)
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
                        "BLEU-1 {} ROUGE_L {} METEOR {}.\n".format(epoch_idx, valid_time, valid_loss,
                                                                             metric_dict["Bleu_1"],
                                                                             metric_dict["ROUGE_L"],
                                                                             metric_dict["METEOR"]))

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

        if best_valid_meteor is None or best_valid_meteor[1] < metric_dict["METEOR"]:
            best_valid_meteor = (epoch_idx, metric_dict["METEOR"])
            save = True

        if save:
            saved_path = os.path.join(config["saved_dir"], config["model"], str(epoch_idx))
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            # save pretrained language model
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(saved_path)
            tokenizer.save_pretrained(saved_path)
            logger.info("Save fine-tuned ELMER model into {}.".format(saved_path))

    logger.info("\n\nThe best loss {} in epoch {}, the best BLEU-1 {} in epoch {}, "
                "the best ROUGE_L {} in epoch {}, the best METEOR {} in epoch {}.\n".format(best_valid_loss[1],
                                                                                            best_valid_loss[0],
                                                           best_valid_bleu[1], best_valid_bleu[0],
                                                           best_valid_rouge[1], best_valid_rouge[0],
                                                           best_valid_meteor[1], best_valid_meteor[0]))


def test(config):
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    logger.info("Load fine-tuned NAR model from {}.".format(config["finetuned_model_dir"]))
    
    tokenizer = ElmerTokenizer.from_pretrained(config["finetuned_model_dir"])
    model = ElmerForConditionalGeneration.from_pretrained(config["finetuned_model_dir"]).to(device)
    model.train(False)

    logger.info("Create testing dataset from {}.".format(config["data_dir"]))
    test_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], tokenizer=tokenizer, data_usage="test"),
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn_seq2seq,
        pin_memory=True)

    calculator = Evaluate()
    model.eval()
    inference_time = []
    generated_text = []
    reference_text = []
    with torch.no_grad():
        for batch in test_dataloader:
            source_input_ids, source_mask, target_input_ids, target_mask, labels = batch

            source_input_ids = source_input_ids.to(device)
            source_mask = source_mask.to(device)
            target_input_ids = target_input_ids.to(device)
            target_mask = target_mask.to(device)
            t0 = time.perf_counter()
            output_dict, inter_losses = model(input_ids=source_input_ids,
                                                 attention_mask=source_mask,
                                                 decoder_input_ids=target_input_ids,
                                                 decoder_attention_mask=target_mask,
                                                 return_dict=True)
            t1 = time.perf_counter()
            final_logits = output_dict["logits"]
            
            output_ids = final_logits.argmax(dim=-1)
            output_sentences = tokenizer.batch_decode(output_ids)
            for sentence in output_sentences:
                try:
                    end = sentence.index("</s>")
                    text = sentence[:end]
                except ValueError:
                    text = sentence
                generated_text.append(text)
                
            reference = tokenizer.batch_decode(labels, skip_special_tokens=True)
            reference_text.extend(reference)
            inference_time.append((t1 - t0) * 1000)

        assert len(generated_text) == len(reference_text)
        
        # remove repetitive generated tokens
        candidate_text = []
        for text in generated_text:
            tokens = []
            for token in text.split():
                if len(tokens) == 0 or token != tokens[-1]:
                    tokens.append(token)
            candidate_text.append(" ".join(tokens))

        saved_file_path = os.path.join(config["output_dir"], config["model"])
        if not os.path.exists(saved_file_path):
            os.makedirs(saved_file_path)
        saved_file = os.path.join(saved_file_path, "output_text.res")
        fout = open(saved_file, "w")
        for i in range(len(candidate_text)):
            fout.write("Generated text: " + candidate_text[i].strip() + "\n")
            fout.write("Reference text: " + reference_text[i].strip() + "\n")
        fout.close()

        candidate_text = [text.lower().strip() for text in candidate_text]
        reference_text = [text.lower().strip() for text in reference_text]
        metric_dict = calculator.evaluate(candidate_text, reference_text)
        avg_time = sum(inference_time) / len(inference_time)
        metric_dict["inference_latency"] = avg_time

        metric_file = os.path.join(saved_file_path, "metric.res")
        fout = open(metric_file, "w")
        fout.write(json.dumps(metric_dict) + "\n")
        fout.close()
        

def main():
    config = read_configuration("config.yaml")

    if config["train"]:
        train(config)
    else:
        test(config)


if __name__ == '__main__':
    main()

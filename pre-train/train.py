import os
import torch
import time
import numpy as np
from logging import getLogger
from data import S2SDataset
from utils import build_optimizer, init_seed, init_logger, read_configuration, collate_fn_seq2seq, format_time
from transformers import BartConfig as ElmerConfig
from transformers import BartTokenizer as ElmerTokenizer
from transformers import BartForConditionalGeneration as ElmerForConditionalGeneration
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm


def train(config):
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])

    logger.info("Initialize ELMER....")
    configuration = ElmerConfig.from_pretrained(config["pretrained_model_dir"])
    model = ElmerForConditionalGeneration(config=configuration).cuda()
    optimizer = build_optimizer(config, model)

    steps = 0
    batch_loss = 0
    cur_batches = 0
    t0 = time.time()
    for epoch_idx in range(config["start_epoch"], config["epochs"]):
        model.train()
        
        logger.info("Create training dataset for Epoch-{} from {}.".format(epoch_idx, config["data_dir"]))
        train_dataloader = DataLoader(
            S2SDataset(data_dir=config["data_dir"], epoch=epoch_idx),
            batch_size=config["train_batch_size"],
            shuffle=False,
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn_seq2seq,
            pin_memory=True)

        for batch_idx, batch in enumerate(train_dataloader):

            # batch data
            source_input_ids, source_mask, target_input_ids, target_mask, labels, exit_layers = batch

            source_input_ids = source_input_ids.cuda()
            source_mask = source_mask.cuda()
            target_input_ids = target_input_ids.cuda()
            target_mask = target_mask.cuda()
            labels = labels.cuda()
            exit_layers = exit_layers.cuda()
            output_dict = model(input_ids=source_input_ids,
                                attention_mask=source_mask,
                                decoder_input_ids=target_input_ids,
                                decoder_attention_mask=target_mask,
                                labels=labels,
                                exit_layers=exit_layers,
                                return_dict=True)

            # Accumulated Gradient Descent
            loss = output_dict["loss"] / config["accumulation_steps"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"])
            batch_loss += loss.item()
            steps += 1

            if steps % config["accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                logger.info("Batch {} time {}: Accumulated Loss {:.4f}.".format(int(steps / config["accumulation_steps"]),
                                                                                format_time(time.time() - t0),
                                                                                batch_loss))
                # Reset batch loss and timing
                batch_loss = 0
                cur_batches += 1
                t0 = time.time()

                if cur_batches % config["save_batches"] == 0:
                    model.eval()
                    saved_path = os.path.join(config["saved_dir"], config["model"], str(cur_batches))
                    if not os.path.exists(saved_path):
                        os.makedirs(saved_path)
                    # save pre-trained language model
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(saved_path)
                    tokenizer.save_pretrained(saved_path)
                    logger.info("Step {}: save ELMER model into {}.".format(steps, saved_path))
                    model.train()


def main():
    config = read_configuration("config.yaml")
    train(config)


if __name__ == '__main__':
    main()

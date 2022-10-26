import json
import torch
import random
import logging
import os
import datetime
import yaml
import re
import sys
import numpy as np
from torch import optim
from logging import getLogger
from transformers.optimization import Adafactor, AdafactorSchedule
from data import BOS_ID, EOS_ID, PAD_ID, MASK_ID


def build_optimizer(config, model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config["optimizer"].lower() == 'adam':
        optimizer = optim.Adam(parameters, lr=config["lr"])
    elif config["optimizer"].lower() == 'sgd':
        optimizer = optim.SGD(parameters, lr=config["lr"])
    elif config["optimizer"].lower() == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=config["lr"])
    elif config["optimizer"].lower() == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=config["lr"])
    elif config["optimizer"].lower() == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config["lr"])
    elif config["optimizer"].lower() == 'adafactor':
        optimizer = Adafactor(parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=lr)
    else:
        raise ValueError('Received unrecognized optimizer {}.'.format(learner))
    return optimizer


def init_seed(seed, reproducibility):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def init_logger(config):
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    logfilename = '{}-{}-{}.log'.format(config["dataset"], config["model"], get_local_time())
    logfilepath = os.path.join(config["log_dir"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config["state"] is None or config["state"].lower() == 'info':
        level = logging.INFO
    elif config["state"].lower() == 'debug':
        level = logging.DEBUG
    elif config["state"].lower() == 'error':
        level = logging.ERROR
    elif config["state"].lower() == 'warning':
        level = logging.WARNING
    elif config["state"].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        handlers=[fh, sh]
    )

def convert_config_dict(config_dict):
    """This function convert the str parameters to their original type.
    """
    for key in config_dict:
        param = config_dict[key]
        if not isinstance(param, str):
            continue
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, dict, bool)):
                value = param
        except (NameError, SyntaxError, TypeError):
            if isinstance(param, str):
                if param.lower() == "true":
                    value = True
                elif param.lower() == "false":
                    value = False
                else:
                    value = param
            else:
                value = param
        config_dict[key] = value
    return config_dict


def read_configuration(config_file):
    # read configuration from yaml file
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        yaml_config_dict = yaml.load(f.read(), Loader=yaml_loader)

    # read configuration from cmd line
    cmd_config_dict = dict()
    unrecognized_args = []
    if "ipykernel_launcher" not in sys.argv[0]:
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                unrecognized_args.append(arg)
                continue
            cmd_arg_name, cmd_arg_value = arg[2:].split("=")
            if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
            else:
                cmd_config_dict[cmd_arg_name] = cmd_arg_value
    if len(unrecognized_args) > 0:
        logger = getLogger()
        logger.warning('command line args [{}] will not be used in TextBox'.format(' '.join(unrecognized_args)))

    cmd_config_dict = convert_config_dict(cmd_config_dict)
    
    final_config_dict = dict()
    final_config_dict.update(yaml_config_dict)
    final_config_dict.update(cmd_config_dict)

    return final_config_dict


def collate_fn_seq2seq(batch):
    source_input_ids, labels = [], []
    for b in batch:
        source_input_ids.append(b[0])
        labels.append(b[1])

    source_input_ids, source_mask = padding(source_input_ids, pad_idx=PAD_ID)
    target_input_ids, target_mask, labels, exit_layers = nar_padding(labels, bos_idx=BOS_ID, mask_idx=MASK_ID, pad_idx=PAD_ID)

    return source_input_ids, source_mask, target_input_ids, target_mask, labels, exit_layers


def padding(inputs, pad_idx):
    max_len = max([len(inp) for inp in inputs])

    padded_inputs, mask = [], []
    for inp in inputs:
        padded_inputs.append(inp + [pad_idx] * (max_len - len(inp)))
        mask.append([1.] * len(inp) + [0.] * (max_len - len(inp)))

    padded_inputs = torch.as_tensor(padded_inputs, dtype=torch.long)
    mask = torch.as_tensor(mask, dtype=torch.bool)

    return padded_inputs, mask


def nar_padding(labels, bos_idx, mask_idx, pad_idx):
    max_len = max([len(lab)-1 for lab in labels])

    masked_inputs, mask, masked_labels, exit_layers = [], [], [], []
    for lab in labels:
        masked_inputs.append([bos_idx] + [mask_idx] * (max_len - 1))
        mask.append([1.] * max_len)
        masked_labels.append(lab[1:] + [pad_idx] * (max_len - len(lab[1:])))
        exit_layers.append(np.random.randint(low=0, high=6, size=max_len).tolist())

    masked_inputs = torch.as_tensor(masked_inputs, dtype=torch.long)
    mask = torch.as_tensor(mask, dtype=torch.bool)
    masked_labels = torch.as_tensor(masked_labels, dtype=torch.long)
    exit_layers = torch.as_tensor(exit_layers, dtype=torch.long)

    return masked_inputs, mask, masked_labels, exit_layers


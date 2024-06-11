import argparse
import os
from torch.backends import cudnn
from pprint import pprint
from sys import exit
import random
import warnings
import wandb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from framework.handlers import get_model, get_db, get_adapt_method

from framework.domain_adaptation.config_ouda import cfg, cfg_from_file
from framework.dataset.weather_cityscapes_list.weather_cityscapes_sets import (
    get_split as rain_split,
)

from framework.dataset.segmentation_db import Segmentation_db, base_transform
from framework.dataset.buffer_db import Buffer_db

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
cudnn.benchmark = True
cudnn.enabled = True
cudnn.deterministic = False
# torch.set_deterministic(True)
getf = lambda x: next(iter(x))


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Code for domain adaptation (DA) training"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="optional config file",
    )
    return parser.parse_args()


def save_model(model, cfg, trg_set):
    root = cfg.OTHERS.SNAPSHOT_DIR
    set_ = cfg.SCHEME.SOURCE
    if not os.path.exists(root):
        os.makedirs(root)
    fname = f"model_train_{set_}_after_{trg_set}.pth"
    torch.save(model.state_dict(), os.path.join(root, fname))


def main():
    # LOAD ARGS
    args = get_arguments()
    print("Called with args:")
    print(args)

    assert args.cfg is not None, "Missing cfg file"
    cfg_from_file(args.cfg)
    # auto-generate snapshot path if not specified
    if cfg.OTHERS.SNAPSHOT_DIR == "":
        os.makedirs(cfg.OTHERS.SNAPSHOT_DIR, exist_ok=True)
    print("Using config:")
    cfg.device = cfg.OTHERS.DEVICE
    pprint(cfg)

    wandb.init(
        project="OUDA",
        config=cfg,
    )
    # INIT
    _init_fn = None
    # fixing seeds
    def _init_fn(seed):
        def init_f(worker_id):
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)
            torch.cuda.manual_seed(seed + worker_id)
            random.seed(seed + worker_id)

        return init_f

    _init_fn(cfg.TRAINING.RANDOM_SEED)(0)
    datasets = get_db(cfg)
    cfg.classnum_to_label = datasets["db_info"]["classnum_to_label"]
    _init_fn(cfg.TRAINING.RANDOM_SEED)(0)
    model = get_model(cfg, len(datasets["db_info"]["label"]))
    cfg.NUM_CLASSES = len(datasets["db_info"]["label"])
    print("Model has been Loaded")

    # Perform source training
    # creating source dataloader
    db_mean = (
        datasets["db_info"]["mean"]
        if cfg.SCHEME.MEAN is None or cfg.SCHEME.MEAN == {}
        else cfg.SCHEME.MEAN
    )
    db_std = (
        datasets["db_info"]["std"]
        if cfg.SCHEME.MEAN is None or cfg.SCHEME.MEAN == {}
        else cfg.SCHEME.STD
    )
    transform = base_transform(np.array(db_mean), np.array(db_std))
    prediction_saving_location = "no_save"
    if cfg.METHOD.ADAPTATION.NAME != {}:
        tmp = cfg.METHOD.ADAPTATION[cfg.METHOD.ADAPTATION.NAME].PREDICTION_SAVE
        prediction_saving_location = tmp if tmp != {} else "no_save"
    original_image = not (
        cfg.SCHEME.ORIGINAL_RES == {}
        or cfg.SCHEME.ORIGINAL_RES == cfg.SCHEME.RESOLUTION
    )
    ds_template = lambda x, dir_str: Segmentation_db(
        cfg.SCHEME.PATH,
        x,
        dict(datasets["db_info"]["label2train"]),
        cfg.SCHEME.RESOLUTION,
        transforms=transform,
        predictions_path=f"{prediction_saving_location}/" + dir_str,
        original_label=original_image,
    )
    dl_template = lambda x, shuffle, dir_str: DataLoader(
        ds_template(x, dir_str),
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.OTHERS.NUM_WORKERS,
        worker_init_fn=_init_fn(cfg.TRAINING.RANDOM_SEED),
    )
    src_train = pd.concat(
        [next(iter(db["train"].values())) for db in datasets["domains_src"]]
    )
    _init_fn(cfg.TRAINING.RANDOM_SEED)(0)
    source_dataloader = {"src": dl_template(src_train, cfg.TRAINING.SHUFFLE, "source")}
    source_val_dataloader = {}
    validation_sets = {}
    if "val" in datasets["domains_src"][0].keys():
        source_val_dataloader = {
            getf(dom["val"].keys()): dl_template(
                getf(dom["val"].values()), False, "source_val"
            )
            for dom in datasets["domains_src"]
        }
        # Evaluation
        validation_sets = source_val_dataloader
        for trg_domain in datasets["domains_trg"]:
            set_ = getf(trg_domain["train"].keys())
            data_val = getf(trg_domain["val"].values())
            val_loader = dl_template(data_val, False, f"trg_val_{set_}")
            validation_sets[set_] = val_loader

    # Testing
    if cfg.METHOD.PRETRAIN.NAME == "EVALUATION":
        from framework.domain_adaptation.methods.adaptation_model import evaluation

        cfg_spec = cfg.METHOD.PRETRAIN["EVALUATION"]
        evaluation_model = evaluation(model, cfg, cfg_spec)
        if "PREDICTION_SAVE" in cfg_spec:
            wandb.run.name = "PRED " + cfg.OTHERS.SNAPSHOT_DIR.split("/")[-1]
            wandb.run.save()
            for trg_domain in datasets["domains_trg"]:
                set_ = getf(trg_domain["train"].keys())
                _init_fn(cfg.TRAINING.RANDOM_SEED)(0)
                data_tr = getf(trg_domain["train"].values())
                trg_loader = dl_template(data_tr, False, f"trg_{set_}")
                cfg_spec.set_ = set_
                evaluation_model.update_cfg_spec(cfg_spec)
                evaluation_model.run_predictions(trg_loader)
        else:
            wandb.run.name = "EVAL " + cfg.OTHERS.SNAPSHOT_DIR.split("/")[-1]
            wandb.run.save()

            log = evaluation_model.evaluate_all(validation_sets)
            log.update(evaluation_model.test_on_samples(validation_sets))
            wandb.log(log)
        exit()

    # Source Training
    if cfg.METHOD.PRETRAIN.NAME == "SEGMENT":
        from framework.domain_adaptation.methods.segmentation import (
            train as train_segment,
        )

        train_segment(
            model,
            source_dataloader,
            source_val_dataloader,
            cfg,
            cfg.METHOD.PRETRAIN.SEGMENT,
        )
        save_model(model, cfg, "src_training")

    # UDA TRAINING
    buff_size = cfg.TRAINING.REPLAY_BUFFER
    if type(buff_size) == float:
        src_sample = src_train.sample(
            frac=buff_size, random_state=cfg.TRAINING.RANDOM_SEED
        )
    else:
        src_sample = src_train.sample(
            n=buff_size, random_state=cfg.TRAINING.RANDOM_SEED
        )
    update_freq = cfg.TRAINING.PERC_FILL_PER_DOMAIN
    if buff_size == 0:
        src_loader = []
    elif isinstance(cfg.TRAINING.BUFFER_DYNAMIC, bool) and cfg.TRAINING.BUFFER_DYNAMIC:
        src_loader = Buffer_db(
            ds_template(src_sample, "source"), cfg.TRAINING.BATCH_SIZE
        )
        print(f"Buffer size: {src_loader.__sizeof__()/(1024**2)} MB")
    else:
        _init_fn(cfg.TRAINING.RANDOM_SEED)(0)
        src_loader = dl_template(src_sample, True, "source")
    # x = Buffer_db(ds_template(src_sample,'source' ), 4)
    print("Starting UDA")

    # Creating a dictionary with all validation
    f_domain = False
    cfg_spec = cfg.METHOD.ADAPTATION[cfg.METHOD.ADAPTATION.NAME]
    da_model = get_adapt_method(cfg)(model, cfg, cfg_spec)
    for order, trg_domain in enumerate(datasets["domains_trg"]):
        set_ = getf(trg_domain["train"].keys())
        _init_fn(cfg.TRAINING.RANDOM_SEED)(0)
        data_tr = getf(trg_domain["train"].values())
        if cfg.TRAINING.SHUFFLE == {} or cfg.TRAINING.SHUFFLE:
            trg_loader = dl_template(data_tr, True, f"trg_{set_}")
        else:
            trg_loader = dl_template(data_tr, False, f"trg_{set_}")
        validation_method = cfg.OTHERS.VALIDATION
        if validation_method == "all":
            val_set = validation_sets
        elif validation_method == "single":
            data_val = getf(trg_domain["val"].values())
            val_set = dl_template(data_val, False, f"trg_val_{set_}")
        elif validation_method == "none":
            val_set = {}
        else:
            raise ValueError(
                f"cfg.OTHERS.VALIDATION value error, it is given {cfg.OTHERS.VALIDATION}"
            )
        cfg_spec.set_ = set_
        if cfg.SCHEME.DOMAIN_OPTIONS != {}:
            if str(set_) in cfg.SCHEME.DOMAIN_OPTIONS:
                for key, value in cfg.SCHEME.DOMAIN_OPTIONS[str(set_)].items():
                    print(f"Selecting values for domain {key}:{value}")
                    cfg_spec[key] = value
        if cfg.SCHEME.ORDER_OPTIONS != {}:
            if order in cfg.SCHEME.ORDER_OPTIONS:
                for key, value in cfg.SCHEME.ORDER_OPTIONS[order].items():
                    print(f"Selecting values for domain {key}:{value}")
                    cfg_spec[key] = value
        cfg_spec.SKIP_CALC |= f_domain
        f_domain = True
        da_model.update_cfg_spec(cfg_spec)
        da_model.train(src_loader, trg_loader, val_set)


if __name__ == "__main__":
    main()

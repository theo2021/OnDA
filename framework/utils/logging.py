import wandb
import json


def wandb_image(sample, pred, label, cfg, caption):
    if "classnum_to_label" in cfg.keys():
        class_labels = cfg.classnum_to_label
        real_image = sample.transpose((1, 2, 0))[:, :, ::-1] * 255
    else:
        db_info = json.load(open(cfg.TRAIN.INFO_TARGET, "r"))
        real_image = sample.transpose((1, 2, 0)) - cfg.TRAIN.IMG_MEAN
        class_labels = dict(zip(range(len(db_info["label"])), db_info["label"]))
    masks = {
        "predictions": {"mask_data": pred, "class_labels": class_labels},
        "ground_truth": {"mask_data": label, "class_labels": class_labels},
    }
    return wandb.Image(real_image, masks=masks, caption=caption)

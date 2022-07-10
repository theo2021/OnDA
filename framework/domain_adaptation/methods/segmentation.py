import torch
import wandb
import copy
from torch.utils.data import DataLoader
from framework.model.deeplabv2 import get_deeplab_v2
from framework.utils.func import loss_calc, prob_2_entropy, _adjust_learning_rate
from framework.domain_adaptation.config import cfg
from framework.domain_adaptation.eval_UDA import evaluate_model
from framework.domain_adaptation.evaluate import segment_sample
from framework.utils.monitoring import PytorchSpeedMeasure
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import torch.backends.cudnn as cudnn


def train(model, train_loader, validation_loaders, cfg, cfg_spec=None):
    # Create the model and start the training.
    if cfg.DOMAIN_ANALYSIS == {}:
        image_size = cfg.SCHEME.RESOLUTION
        lr = cfg_spec.LEARNING_RATE
        momentum = cfg_spec.MOMENTUM
        epochs = cfg_spec.EPOCHS
        val_multi = 1
        weight_decay = cfg_spec.WEIGHT_DECAY
        samples_every = cfg.OTHERS.GENERATE_SAMPLES_EVERY
        device = cfg.OTHERS.DEVICE
        perf_record = PytorchSpeedMeasure(on=cfg.OTHERS.SCHEDULE)

    else:
        image_size = cfg.DOMAIN_ANALYSIS.DATASET.RESOLUTION
        lr = cfg.LEARNING_RATE
        momentum = cfg.MOMENTUM
        weight_decay = cfg.WEIGHT_DECAY
        epochs = cfg.EPOCHS
        val_multi = cfg.VAL_EPOCHS_MULTI
        samples_every = cfg.GENERATE_SAMPLES_EVERY
        device = cfg.device
        perf_record = PytorchSpeedMeasure(on=True)
    # target_models = [copy.deepcopy(model).to(device).train() for _ in range(len(validation_loaders.keys()))]
    # SEGMNETATION NETWORK
    cudnn.benchmark = True
    cudnn.enabled = True

    interp = torch.nn.Upsample(
        size=image_size[::-1], mode="bilinear", align_corners=True
    )
    if not (cfg.SCHEME.ORIGINAL_RES == {} or cfg.SCHEME.ORIGINAL_RES == image_size):
        interp_original = torch.nn.Upsample(
            size=cfg.SCHEME.ORIGINAL_RES[::-1], mode="bilinear", align_corners=True
        )
    else:
        interp_original = None
    optimizer_template = lambda x: torch.optim.SGD(
        x.optim_parameters(lr), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    optimizer = optimizer_template(model)
    train_loader = next(iter(train_loader.values()))
    step = 0
    avrg = []
    for epoch in range(epochs):
        model.train()
        print(f"Epoch: {epoch}")
        perf_record.reset_timer()
        for batch in tqdm(train_loader):
            perf_record.add("Batch Fetch")
            optimizer.zero_grad()
            perf_record.reset_timer()
            prediction_aux, prediction = model(batch["image"].to(device))
            perf_record.add("Forward Pass")
            if type(prediction) == dict:
                prediction = prediction["out"]
            loss = loss_calc(interp(prediction), batch["label"], device)
            if prediction_aux is not None:
                if type(prediction_aux) == dict:
                    prediction_aux = prediction_aux["out"]
                loss_aux = loss_calc(interp(prediction_aux), batch["label"], device)
                loss = loss + 0.1 * loss_aux
            perf_record.add("Loss Calculation")
            loss.backward()
            perf_record.add("Backward")
            optimizer.step()
            perf_record.add("Update Step")
            _adjust_learning_rate(
                optimizer, step, cfg, lr, len(train_loader) * epochs, cfg_spec.POWER
            )
            perf_record.add("Learning Rate Adjust")
            avrg.append(loss.item())
            if step % 10 == 0:
                wandb.log(
                    {
                        "Segmentation loss": sum(avrg) / len(avrg),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                avrg = []
                if not perf_record.switch_off:
                    print(perf_record.avg())
            step += 1
            perf_record.reset_timer()
        print("Evaluation")
        log = {"epoch": epoch * val_multi}
        for set_, eval_loader in validation_loaders.items():
            if interp_original is not None:
                IoU, entropy, IoU_hard = evaluate_model(
                    model,
                    eval_loader,
                    interp,
                    cfg,
                    return_entropy=True,
                    intrp_org=interp_original,
                )
                log[f"Val mIoU full image of {set_}"] = np.nanmean(IoU_hard)
            else:
                IoU, entropy = evaluate_model(
                    model, eval_loader, interp, cfg, return_entropy=True
                )
            log[f"Val mIoU of {set_}"] = np.nanmean(IoU)
            log[f"Val std IoU of {set_}"] = np.nanstd(IoU)
            log[f"val entropy of {set_}"] = entropy
            iterator = iter(eval_loader)
            if epoch % samples_every == 0:
                samples = []
                for _ in range(10):  # number of examples
                    samples.append(next(iterator))
                for i, sample in enumerate(samples):
                    wimg = segment_sample(
                        model,
                        sample["image"][0],
                        interp,
                        sample["label"][0],
                        cfg,
                        f"Sample from {set_}",
                    )
                    log[f"Condition {set_} sample {i}"] = wimg
        wandb.log(log)
        save_model(model, epoch, cfg)


def save_model(model, epoch, cfg):

    root = cfg.SNAPSHOT_DIR
    set_ = cfg.DOMAIN_ANALYSIS.DATASET.TRAIN
    if root == {}:
        root = cfg.OTHERS.SNAPSHOT_DIR
        set_ = cfg.SCHEME.SOURCE
    if not os.path.exists(root):
        os.makedirs(root)
    fname = f"model_train_{set_}.pth"
    torch.save(model.state_dict(), os.path.join(root, fname))

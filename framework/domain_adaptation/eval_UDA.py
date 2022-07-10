# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------

import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from torch.functional import F
from tqdm import tqdm

from framework.utils.func import per_class_iu, fast_hist, prob_2_entropy
from framework.utils.serialization import pickle_dump, pickle_load


def evaluate_model(
    model, val_loader, interp, cfg, return_entropy=False, intrp_org=None, prototype=None
):
    def transfer(x):
        return x.cuda(cfg.GPU_ID) if "GPU_ID" in cfg else x.to(cfg.device)

    model.eval()
    hist = 0
    hist_org = 0
    entropies = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            mdl_out = model(transfer(batch["image"]))[1]
            if type(mdl_out) == dict:
                if prototype is not None:
                    batches, channels, w, h = mdl_out["out"].size()
                    prior = 1
                    if batch["soft_predictions"] is not None:
                        prior = transfer(batch["soft_predictions"])
                    mdl_out = (
                        prototype.pseudo_labels(mdl_out["feat"], prior=prior, soft=True)
                        .reshape(batches, w, h, channels)
                        .permute(0, 3, 1, 2)
                    )
                else:
                    mdl_out = mdl_out["out"]
            pred_main = interp(mdl_out)
            if return_entropy:
                entropies.append(
                    prob_2_entropy(F.softmax(pred_main, dim=1)).mean().item()
                )
            for prediction, label in zip(pred_main, batch["label"]):
                label = label.numpy()
                pred_labels = prediction.permute(1, 2, 0).argmax(dim=2).cpu().numpy()
                hist += fast_hist(
                    label.flatten(), pred_labels.flatten(), cfg.NUM_CLASSES
                )
            if intrp_org is not None:
                pred_org = intrp_org(mdl_out)
                for prediction, label in zip(pred_org, batch["label_raw"]):
                    label = label.numpy()
                    pred_labels = (
                        prediction.permute(1, 2, 0).argmax(dim=2).cpu().numpy()
                    )
                    hist_org += fast_hist(
                        label.flatten(), pred_labels.flatten(), cfg.NUM_CLASSES
                    )
    model.train()
    out = (per_class_iu(hist),)
    if return_entropy:
        out += (np.mean(entropies),)
    if intrp_org is not None:
        out += (per_class_iu(hist_org),)
    return out


def evaluate_domain_adaptation(
    models, test_loader, cfg, fixed_test_size=True, verbose=True
):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(
            size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
            mode="bilinear",
            align_corners=True,
        )
    # eval
    if cfg.TEST.MODE == "single":
        eval_single(cfg, models, device, test_loader, interp, fixed_test_size, verbose)
    elif cfg.TEST.MODE == "best":
        eval_best(cfg, models, device, test_loader, interp, fixed_test_size, verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models, device, test_loader, interp, fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), "Number of models are not matched"
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(
                size=(label.shape[1], label.shape[2]),
                mode="bilinear",
                align_corners=True,
            )
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.cuda(device))[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, "Output is None"
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
    inters_over_union_classes = per_class_iu(hist)
    print(f"mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}")
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)


def eval_best(cfg, models, device, test_loader, interp, fixed_test_size, verbose):
    assert len(models) == 1, "Not yet supported multi models in this mode"
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), "SNAPSHOT_DIR is not found"
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], "all_res.pkl")
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ""
    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f"model_{i_iter}.pth")
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print("Waiting for model..!")
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(test_loader)
            for index in tqdm(range(len(test_loader))):
                image, label, _, name = next(test_iter)
                if not fixed_test_size:
                    interp = nn.Upsample(
                        size=(label.shape[1], label.shape[2]),
                        mode="bilinear",
                        align_corners=True,
                    )
                with torch.no_grad():
                    pred_main = models[0](image.cuda(device))[1]
                    output = interp(pred_main).cpu().data[0].numpy()
                    output = output.transpose(1, 2, 0)
                    output = np.argmax(output, axis=2)
                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print(
                        "{:d} / {:d}: {:0.2f}".format(
                            index,
                            len(test_loader),
                            100 * np.nanmean(per_class_iu(hist)),
                        )
                    )
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print("\tCurrent mIoU:", computed_miou)
        print("\tCurrent best model:", cur_best_model)
        print("\tCurrent best mIoU:", cur_best_miou)
        if verbose:
            display_stats(
                cfg, test_loader.dataset.class_names, inters_over_union_classes
            )


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(
            name_classes[ind_class]
            + "\t"
            + str(round(inters_over_union_classes[ind_class] * 100, 2))
        )

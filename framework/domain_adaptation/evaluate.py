from framework.utils.func import is_turn
from framework.domain_adaptation.eval_UDA import evaluate_model
from framework.utils.logging import wandb_image
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


def evaluation_record(
    i_iter,
    model,
    trainloader,
    targetloader,
    validation_loaders,
    interp,
    interp_target,
    cfg,
    current_losses,
    models,
):
    device = cfg.GPU_ID
    if is_turn(i_iter, cfg.TRAIN.VALIDATION_STEP):
        mdl_eval_source = evaluate_model(
            model, validation_loaders["source"], interp, cfg
        )
        miou_source = np.nanmean(mdl_eval_source) * 100
        tqdm.write(f"Source mIoU = \t{round(miou_source, 2)}")
        tqdm.write("Per class: {}".format(mdl_eval_source))
        mdl_eval_target = evaluate_model(
            model, validation_loaders["target"], interp_target, cfg
        )
        miou_target = np.nanmean(mdl_eval_target) * 100
        tqdm.write(f"Target mIoU = \t{round(miou_target, 2)}")
        tqdm.write("Per class: {}".format(mdl_eval_target))
        current_losses["mIoU_src"] = miou_source
        current_losses["mIoU_trg"] = miou_target

    if is_turn(i_iter, cfg.TRAIN.RECORD_IMAGES_EVERY):
        data = [
            trainloader.dataset[cfg.TRAIN.SRC_TRAIN_IMAGE_INDEX],
            targetloader.dataset[cfg.TRAIN.TRG_TRAIN_IMAGE_INDEX],
            validation_loaders["source"].dataset[cfg.TRAIN.SRC_VAL_IMAGE_INDEX],
            validation_loaders["target"].dataset[cfg.TRAIN.TRG_VAL_IMAGE_INDEX],
        ]
        samples, labels = list(
            zip(*[[sample["image"], sample["label"]] for sample in data])
        )  # list(zip(*data))[:2]
        samples = [torch.from_numpy(sample) for sample in samples]
        captions = [
            "Source Train",
            "Target Train",
            "Source Validation",
            "Target Validation",
        ]
        model.eval()
        examples_source = []
        examples_target = []
        with torch.no_grad():
            for i, (sample, label, caption) in enumerate(
                zip(samples, labels, captions)
            ):
                prediction = model(sample.unsqueeze(0).cuda(device))[1]
                if i % 2 == 0:
                    prediction = interp(prediction)
                    prediction = prediction.squeeze(0).argmax(dim=0).cpu().numpy()
                    examples_source.append(
                        wandb_image(
                            sample.cpu().numpy(), prediction, label, cfg, caption
                        )
                    )
                else:
                    prediction = interp_target(prediction)
                    prediction = prediction.squeeze(0).argmax(dim=0).cpu().numpy()
                    examples_target.append(
                        wandb_image(
                            sample.cpu().numpy(), prediction, label, cfg, caption
                        )
                    )
        current_losses["step {} Source".format(str(i_iter))] = examples_source
        current_losses["step {} Target".format(str(i_iter))] = examples_target
        model.train()

    if is_turn(i_iter, cfg.TRAIN.SAVE_PRED_EVERY):
        print("taking snapshot ...")
        print("exp =", cfg.TRAIN.SNAPSHOT_DIR)
        save_model(model, "", i_iter, cfg)
        for name, mdl in models.items():
            save_model(mdl, name, i_iter, cfg)

    if is_turn(i_iter, cfg.TRAIN.MEASURE_PREDICTION_TIME):
        model.eval()
        start.record()
        with torch.no_grad():
            for batch in validation_loaders["target"]:
                prediction = interp(model(batch["image"].cuda(device))[1])
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        current_losses["prediction_time_val"] = time / len(validation_loaders["target"])
        model.train()


def save_model(model, name, i_iter, cfg):
    snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
    torch.save(model.state_dict(), snapshot_dir / f"model_{i_iter}_{name}.pth")


def segment_sample(model, sample, interp, label, cfg, caption):
    model.eval()
    with torch.no_grad():
        pred = model(sample.unsqueeze(0).to(cfg.device))[1]
        if type(pred) == dict:
            pred = pred["out"]
        pred = interp(pred).squeeze(0).argmax(dim=0).cpu().numpy()
    model.train()
    return wandb_image(sample.cpu().numpy(), pred, label.numpy(), cfg, caption)

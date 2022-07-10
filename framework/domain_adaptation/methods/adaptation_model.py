import os
import abc
from copy import deepcopy
from statistics import mode
from pathlib import Path
import torch
import numpy as np
from torch import functional
from tqdm import tqdm
from torch import nn, optim
from torch.functional import F
from torch.backends import cudnn
import wandb
from framework.utils.func import (
    loss_calc,
    lr_poly,
    fast_hist,
    per_class_iu,
    prob_2_entropy,
    bce_loss,
)
from framework.utils.loss import rce, js_divergance
from framework.utils.monitoring import Monitor, PytorchSpeedMeasure, ECE
from framework.domain_adaptation.evaluate import segment_sample
from framework.domain_adaptation.methods.prototype_handler import prototype_handler
from framework.model.discriminator import get_fc_discriminator


def switch_batch_statistics(model, setting):
    """Frezes and defreezes all batch norm update from a model"""
    assert isinstance(
        setting, bool
    ), f"setting value should be a boolean, given: {setting}"
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = setting


class batchnorm_stats:
    def __init__(self, model) -> None:
        self.memory = {}
        self.model = model
        self.save()

    def compare(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                print(
                    {
                        "model": module.running_mean,
                        "memory": self.memory[module_name]["running_mean"],
                        "diff": module.running_mean
                        - self.memory[module_name]["running_mean"],
                    }
                )

    def save(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.memory[module_name] = deepcopy(module.state_dict())

    def load(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.load_state_dict(self.memory[module_name])

    def exchange(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                current_state = deepcopy(module.state_dict())
                module.load_state_dict(self.memory[module_name])
                self.memory[module_name] = current_state


class da_model:
    def __init__(self, model, cfg, cfg_spec) -> None:
        self.model = model
        self.bn = batchnorm_stats(model)
        self.cfg = cfg
        self.cfg_spec = cfg_spec  # specific model configurations
        self.device = cfg.OTHERS.DEVICE
        input_size_source = cfg.SCHEME.RESOLUTION
        learning_rate = cfg_spec.LEARNING_RATE
        cudnn.deterministic = True
        cudnn.enabled = True
        cudnn.benchmark = False
        # torch.set_deterministic(True)
        self.optimizer = optim.SGD(
            model.optim_parameters(learning_rate),
            lr=learning_rate,
            momentum=cfg_spec.MOMENTUM,
            weight_decay=cfg_spec.WEIGHT_DECAY,
        )
        self.interp = nn.Upsample(
            size=(input_size_source[1], input_size_source[0]),
            mode="bilinear",
            align_corners=True,
        )
        self.eval_metric_list = []
        self.ece_record = not (
            isinstance(cfg.OTHERS.ECE_SKIP, bool) and cfg.OTHERS.ECE_SKIP
        )
        self.prediction_counter = {}

    @abc.abstractmethod
    def models_eval(self):
        pass

    @abc.abstractmethod
    def models_default_config(self):
        pass

    def update_cfg_spec(self, new_cfg):
        self.cfg_spec = new_cfg

    def adjust_learning_rate(self, step, total_steps):
        if self.cfg.MODEL.LR_RATIO is None or self.cfg.MODEL.LR_RATIO == {}:
            self.cfg.MODEL.LR_RATIO = "1:10"
        ratios = [int(v) for v in self.cfg.MODEL.LR_RATIO.split(":")]
        learning_rate = lr_poly(
            self.cfg_spec.LEARNING_RATE, step, total_steps, self.cfg_spec.POWER
        )
        self.optimizer.param_groups[0]["lr"] = learning_rate * ratios[0]
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]["lr"] = learning_rate * ratios[1]

    def evaluate(self, validation_loader, additional_func={}):
        """Evaluates the current model"""
        function_dict = {
            "model": lambda x: self.model(x["image"].to(self.device))[1]["out"]
        }
        function_dict.update(additional_func)

        bins = 1000 if self.cfg.OTHERS.BINS == {} else self.cfg.OTHERS.BINS
        if self.ece_record:
            ece_recorder = {}
            for key in function_dict:
                ece_recorder[key] = ECE(bins)
        self.models_eval()
        counters = {key: 0 for key in function_dict}
        with torch.no_grad():
            for batch in tqdm(validation_loader):
                for key, func in function_dict.items():
                    pred = func(batch)
                    pred = self.interp(pred).softmax(axis=1)
                    if self.ece_record:
                        ece_recorder[key].record(
                            pred, batch["label"].to(self.device), axis=1
                        )
                    for item_pred, label in zip(pred, batch["label"]):
                        label = label.numpy()
                        item_pred_labels = (
                            item_pred.permute(1, 2, 0).argmax(dim=2).cpu().numpy()
                        )
                        counters[key] += fast_hist(
                            label.flatten(),
                            item_pred_labels.flatten(),
                            self.cfg.NUM_CLASSES,
                        )
        self.models_default_config()
        if self.ece_record:
            self.eval_metric_list.extend(
                [("ece " + name, ece().item()) for name, ece in ece_recorder.items()]
            )
            ece_recorder = {}
        return {key: per_class_iu(count) for key, count in counters.items()}

    def evaluate_all(self, validation_loaders):
        """Evaluate a set of dataloaders"""
        validation_log = {}
        for val_set, val_loader in validation_loaders.items():
            result = self.evaluate(val_loader)
            for key, value in result.items():
                validation_log[f"Val mIoU {key} of {val_set}"] = np.nanmean(value)
                validation_log[f"Val std IoU {key} of {val_set}"] = np.nanstd(value)
            for name, value in self.eval_metric_list:
                validation_log[f"{name} {val_set}"] = value
            self.eval_metric_list = []
        return validation_log

    def test_on_samples(self, validation_loaders):
        self.models_eval()
        log = {}
        for val_set, val_loader in validation_loaders.items():
            samples = []
            tmp_iter = iter(val_loader)
            for _ in range(10):  # number of examples
                samples.append(next(tmp_iter))
            for i, sample in enumerate(samples):
                wimg = segment_sample(
                    self.model,
                    sample["image"][0],
                    self.interp,
                    sample["label"][0],
                    self.cfg,
                    f"Sample from {val_set}",
                )
                log[f"Condition {val_set} sample {i}"] = wimg
        self.models_default_config()
        return log

    def save_model(self, model_dict=None, prefix=""):
        if model_dict is None:
            model_dict = {"model": self.model}
        root = self.cfg.OTHERS.SNAPSHOT_DIR
        set_ = self.cfg_spec.set_
        if not os.path.exists(root):
            os.makedirs(root)
        for key, model in model_dict.items():
            fname = f"{key}_{prefix}.pth"
            # if not os.path.exists(os.path.join(root, fname)):
            torch.save(model.state_dict(), os.path.join(root, fname))

    def load_model(self, path):
        print(f"Model {path} is being loaded")
        self.model.load_state_dict(torch.load(path))

    def save_prediction(self, prediction):
        base_path = os.path.join(
            self.cfg_spec.PREDICTION_SAVE, "_".join(str(self.cfg_spec.set_))
        )
        if self.cfg_spec.set_ not in self.prediction_counter:
            self.prediction_counter[self.cfg_spec.set_] = 0
            if not os.path.exists(base_path):
                os.makedirs(base_path)
        torch.save(
            prediction,
            os.path.join(
                base_path, f"batch-{self.prediction_counter[self.cfg_spec.set_]}.pt"
            ),
        )
        self.prediction_counter[self.cfg_spec.set_] += 1

    def run_predictions(self, trg_loader):
        self.models_eval()
        with torch.no_grad():
            for i, batch in enumerate(trg_loader):
                prediction_batch = self.model(batch["image"].to(self.device))
                confidence = (
                    prediction_batch[1]["out"].softmax(axis=1).max(axis=1)[0].mean()
                )
                wandb.log(
                    {
                        "Prediction confidence": confidence,
                        "Progress": i * 100.0 / len(trg_loader),
                    }
                )
                self.save_prediction(prediction_batch[1]["out"].cpu())
        self.models_default_config()


class evaluation(da_model):
    def __init__(self, model, cfg, cfg_spec) -> None:
        super().__init__(model, cfg, cfg_spec)
        dirpath = self.cfg.OTHERS.SNAPSHOT_DIR
        if dirpath != "NONE":
            paths = sorted(Path(dirpath).iterdir(), reverse=True, key=os.path.getmtime)
            latest_save = [path for path in paths if "pth" in str(path)][0]
            super().load_model(latest_save)

    def models_eval(self):
        self.model.eval()

    def models_default_config(self):
        self.model.eval()

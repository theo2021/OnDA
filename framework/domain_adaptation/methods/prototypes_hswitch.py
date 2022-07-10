import os
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.functional import F
from torch.backends import cudnn
import wandb
from framework.utils.func import loss_calc, lr_poly, fast_hist, per_class_iu
from framework.utils.loss import rce, js_divergance
from framework.utils.monitoring import Monitor
from framework.domain_adaptation.evaluate import segment_sample
from framework.domain_adaptation.methods.prototype_handler import prototype_handler
from framework.domain_adaptation.methods.prototypes import (
    online_proDA,
    switch_batch_statistics,
)


class hswitch_proDA(online_proDA):
    def __init__(self, model, cfg, cfg_spec) -> None:
        # Creating the models
        super(hswitch_proDA, self).__init__(model, cfg, cfg_spec)

    def prototype_predictions(self, batch):
        """From batch calculate Target prototype predictions"""
        with torch.no_grad():
            batch_image = batch["image"].to(self.device)
            # EMA Model
            _, pred_trg_ema_main = self.ema_model(batch_image)
            prior_ema = pred_trg_ema_main["out"].softmax(axis=1)
            self.intensity_ma.add({"prior EMA": prior_ema.max(axis=1)[0].mean()})
            prior = self.cfg_spec.EMA_LAMBDA * prior_ema
            self.record_ece("ema", prior_ema, batch["label"])
            # Static Model
            if self.cfg_spec.STATIC_LAMBDA > 0:
                _, pred_trg_first_main = self.static_model(batch_image)
                prior_static = pred_trg_first_main["out"].softmax(axis=1)
                self.intensity_ma.add(
                    {"prior static": prior_static.max(axis=1)[0].mean()}
                )
                prior += self.cfg_spec.STATIC_LAMBDA * prior_static
                self.record_ece("static", prior_static, batch["label"])
            if self.cfg_spec.SOFT_TRANS:
                # dyn_regularizer = max(0.965, self.intensity_ma.avg('prior dynamic'))/0.965
                vl = self.intensity_ma.avg("prior static")
                percentage_static = max(min(vl * (25.0 / 3) - (41.0 / 6), 1), 0)
            else:
                percentage_static = int(
                    self.intensity_ma.avg("prior static")
                    > self.cfg_spec.SWITCH_PRIOR_THRESH
                )
            self.intensity_ma.add({"percentage_static": percentage_static})
            prior *= percentage_static
            # Dynamic Model
            if self.cfg_spec.DYNAMIC_LAMBDA > 0 and percentage_static < 1:
                _, pred_trg_dynamic_main = self.dynamic_model(batch_image)
                prior_dynamic = pred_trg_dynamic_main["out"].softmax(axis=1)
                self.record_ece("dynamic", prior_dynamic, batch["label"])
                self.intensity_ma.add(
                    {"prior dynamic": prior_dynamic.max(axis=1)[0].mean()}
                )
                prior += (
                    (1 - percentage_static)
                    * self.cfg_spec.DYNAMIC_LAMBDA
                    * prior_dynamic
                )
        feat = pred_trg_ema_main["feat"]
        # for the pseudolabels we use the feature representation from ema
        # together with the predictions before adaptation
        self.intensity_ma.add({"prior": prior.max(axis=1)[0].mean()})
        pseudolabels = self.prototypes.pseudo_labels(
            feat, prior, confidence_monitor=self.intensity_ma
        )
        # calculation also soft predictions to record confidence
        soft_predictions = self.prototypes.pseudo_labels(feat, prior, soft=True)
        self.intensity_ma.add(
            {"pseudolabel confidence": soft_predictions.max(axis=1)[0].mean()}
        )
        return {
            "ema_model": pred_trg_ema_main,
            "pseudolabels": pseudolabels,
            "soft_predictions": soft_predictions,
        }

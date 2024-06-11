from torch.nn.functional import threshold
from framework.domain_adaptation.methods.adaptation_model import da_model
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
from framework.utils.monitoring import PytorchSpeedMeasure
from framework.utils.loss import rce, js_divergance
from framework.utils.monitoring import Monitor, ECE
from framework.domain_adaptation.evaluate import segment_sample
from framework.domain_adaptation.methods.prototype_handler import prototype_handler
from framework.domain_adaptation.methods.adaptation_model import (
    da_model,
    switch_batch_statistics,
)
from framework.utils.ewc import ewc_loss

cudnn.deterministic = False
cudnn.enabled = True
cudnn.benchmark = True
# torch.set_deterministic(True)


def regular_loss(regularizer, activation):
    loss = 0
    logp = F.log_softmax(activation, dim=1)
    if regularizer == "MRENT":
        p = F.softmax(activation, dim=1)
        loss = (p * logp).sum() / (p.shape[0] * p.shape[2] * p.shape[3])
    elif regularizer == "MRKLD":
        loss = -logp.sum() / (
            logp.shape[0] * logp.shape[1] * logp.shape[2] * logp.shape[3]
        )
    return loss


class online_proDA(da_model):
    def __init__(self, model, cfg, cfg_spec) -> None:
        # Creating the models
        super(online_proDA, self).__init__(model, cfg, cfg_spec)
        self.ema_model = deepcopy(model)
        self.dynamic_model = deepcopy(model)
        self.static_model = deepcopy(model)
        args = [cfg_spec.AVG_MONITOR_SIZE]
        if cfg_spec.EXP_MONITOR_CONST != {}:
            args.append(cfg_spec.EXP_MONITOR_CONST)
        if cfg_spec.DEV_MONITOR_FUNC != {}:
            args.append(cfg_spec.DEV_MONITOR_FUNC)
        self.intensity_ma = Monitor(*args)
        for module in self.static_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = cfg_spec.BN_MOMENTUM
        self.models_default_config()
        # Configurations
        self.prototypes = prototype_handler(
            ma_lambda=cfg_spec.MA_LAMBDA,
            tau=cfg_spec.TAU,
            thresh=cfg_spec.PSEUDO_THRESH,
            distance_metric=cfg_spec.DISTANCE_MEASURE,
            confidence_regularization_threshold=cfg_spec.CONFIDENCE_REGULARIZATION_THRESHOLD,
        )
        self.skip_proto = False
        if isinstance(cfg_spec.LOAD_PROTO, str):
            self.prototypes.load(cfg_spec.LOAD_PROTO)
            self.skip_proto = True
        self.proto_loc = cfg.OTHERS.SNAPSHOT_DIR + f"/proto_{cfg_spec.set_}.pickle"
        self.proto_cur = cfg.OTHERS.SNAPSHOT_DIR + "/proto_current.pickle"
        self.probability_per_step = (
            0
            if cfg.TRAINING.PERC_FILL_PER_DOMAIN == {}
            else cfg.TRAINING.PERC_FILL_PER_DOMAIN
        )
        self.probability_per_step *= (
            1.0 * cfg.TRAINING.REPLAY_BUFFER / cfg.TRAINING.BATCH_SIZE
        )
        if (
            self.cfg_spec.MODEL_REGULARIZATION != {}
            and self.cfg_spec.MODEL_REGULARIZATION > 0
        ):
            self.model_regularization = lambda x: ewc_loss(
                self.cfg_spec.MODEL_REGULARIZATION,
                list(self.static_model.parameters()),
                list(x),
            )
        else:
            self.model_regularization = None
        self.ece_save = {}
        if isinstance(cfg_spec.BN_POLICY, dict):  # freeze, double, static
            self.cfg_spec.BN_POLICY = "freeze"
        if cfg_spec.LOAD_MODEL != {} and cfg_spec.LOAD_MODEL:
            super().load_model(cfg_spec.LOAD_MODEL)
        self.dynamic_update_counter = 0
        self.speed_monitor = PytorchSpeedMeasure()

    def update_dynamic(self):
        """Update the dynamic model"""
        self.dynamic_model = deepcopy(self.model)
        self.models_default_config()

    def models_default_config(self):
        """Default model configurations"""
        self.model.train()
        self.ema_model.train()
        self.dynamic_model.eval()
        self.static_model.eval()
        self.intensity_ma.train()

    def models_eval(self):
        """Set all models to eval mode"""
        self.model.eval()
        self.ema_model.eval()
        self.dynamic_model.eval()
        self.static_model.eval()
        self.intensity_ma.eval()

    def update_cfg_spec(self, new_cfg):
        super().update_cfg_spec(new_cfg)
        self.proto_loc = self.cfg.OTHERS.SNAPSHOT_DIR + f"/proto_{new_cfg.set_}.pickle"

    def save_model(self):
        super().save_model(prefix="current")
        self.prototypes.save(self.proto_loc)

    def calculate_prototypes(self, dataloader):
        """Compute the prototypes from a dataloader"""
        with torch.no_grad():
            if (
                isinstance(self.cfg.TRAINING.BUFFER_DYNAMIC, bool)
                and self.cfg.TRAINING.BUFFER_DYNAMIC
            ):
                l_loader = dataloader.sequential()
            else:
                l_loader = dataloader
            for batch_target in tqdm(l_loader):
                _, pred_trg_main = self.model(batch_target["image"].to(self.device))
                feat = pred_trg_main["feat"]
                out = pred_trg_main["out"]
                if self.cfg_spec.STARTING_PROTO == "source":
                    _, channels, height, width = out.size()
                    labels_clone = F.interpolate(
                        batch_target["label"].unsqueeze(1).float(), size=(height, width)
                    ).view(-1)
                    mask = (labels_clone != 255).to(self.device)
                    feat_clone = feat.permute(1, 0, 2, 3).reshape(feat.shape[1], -1)
                    labels_filtered = labels_clone[mask]
                    feat = feat_clone[:, mask].permute(1, 0)
                    out = torch.nn.functional.one_hot(
                        labels_filtered.long(), channels
                    ).to(self.device)
                self.prototypes.append(feat, out)
        self.prototypes.save(self.proto_cur)

    def supervised_loss(self, batch):
        """Takes a sample batch and returns the source losses"""
        _, pred_src_main = self.model(batch["image"].to(self.device))
        # feat = pred_src_main['feat'].detach()
        out = pred_src_main["out"]
        total_loss = 0
        ce_loss = 0
        rce_loss = 0
        if "stored_predictions" in batch.keys():
            label = batch["stored_predictions"]
        else:
            label = batch["label_res"]
        if self.cfg_spec.BUFF_CE > 0:
            ce_loss = loss_calc(out, label, self.device)
            total_loss += self.cfg_spec.BUFF_CE * ce_loss
        if self.cfg_spec.BUFF_RCE > 0:
            rce_loss = rce(out, label, self.device)
            total_loss += self.cfg_spec.BUFF_RCE * rce_loss
        # _, channels, height, width = pred_src_main['out'].size()
        # labels_clone = F.interpolate(batch['label'].unsqueeze(1).float(),
        #     size=(height, width)).view(-1)
        # mask = (labels_clone != 255).to(self.device)
        # feat_clone = feat.permute(1,0,2,3).reshape(feat.shape[1], -1)
        # labels_filtered = labels_clone[mask]
        # feat = feat_clone[:, mask].permute(1,0)
        # out = torch.nn.functional.one_hot(labels_filtered.long(), \
        #     channels).to(self.device)
        # self.prototypes.ma(feat, out)
        return {
            "buff_ce_loss": ce_loss,
            "buff_rce_loss": rce_loss,
            "buff_loss": total_loss,
        }

    def record_ece(self, name, prediction, label):
        if self.ece_record:
            name = "ece " + name
            bins = 1000 if self.cfg.OTHERS.BINS == {} else self.cfg.OTHERS.BINS
            if self.intensity_ma.freeze:
                if name not in self.ece_save.keys():
                    self.ece_save[name] = ECE(bins)
                if label.device != prediction.device:
                    label = label.to(prediction.device)
                self.ece_save[name].record(self.interp(prediction), label, axis=1)

    def register_ece(self):
        if self.ece_record:
            for name, ece in self.ece_save.items():
                self.eval_metric_list.append((name, ece().item()))
            self.ece_save = {}

    def prototype_predictions(self, batch):
        """From batch calculate Target prototype predictions"""
        with torch.no_grad():
            batch_image = batch["image"].to(self.device)
            # EMA Model
            _, pred_trg_ema_main = self.ema_model(batch_image)
            prior_ema = pred_trg_ema_main["out"].softmax(axis=1)
            self.intensity_ma.add({"prior EMA": prior_ema.max(axis=1)[0].mean()})
            self.record_ece("ema", prior_ema, batch["label"])
            prior = self.cfg_spec.EMA_LAMBDA * prior_ema
            # Static Model
            if self.cfg_spec.STATIC_LAMBDA > 0:
                _, pred_trg_first_main = self.static_model(batch_image)
                prior_static = pred_trg_first_main["out"].softmax(axis=1)
                self.intensity_ma.add(
                    {"prior static": prior_static.max(axis=1)[0].mean()}
                )
                prior += self.cfg_spec.STATIC_LAMBDA * prior_static
                self.record_ece("static", prior_static, batch["label"])
            # Dynamic Model
            calculate_dyn = True
            replace_dyn = False
            if (
                self.cfg_spec.SWITCH_PRIOR_THRESH > 0
                and self.intensity_ma.avg("prior static")
                < self.cfg_spec.SWITCH_PRIOR_THRESH
            ):
                replace_dyn = True
            elif self.cfg_spec.SWITCH_PRIOR_THRESH > 0:
                calculate_dyn = False
            if self.cfg_spec.DYNAMIC_LAMBDA > 0 and calculate_dyn:
                _, pred_trg_dynamic_main = self.dynamic_model(batch_image)
                prior_dynamic = pred_trg_dynamic_main["out"].softmax(axis=1)
                self.record_ece("dynamic", prior_dynamic, batch["label"])
                self.intensity_ma.add(
                    {"prior dynamic": prior_dynamic.max(axis=1)[0].mean()}
                )
                if replace_dyn:
                    prior = self.cfg_spec.DYNAMIC_LAMBDA * prior_dynamic
                else:
                    prior += self.cfg_spec.DYNAMIC_LAMBDA * prior_dynamic
        feat = pred_trg_ema_main["feat"]
        # for the pseudolabels we use the feature representation from ema
        # together with the predictions before adaptation
        self.intensity_ma.add({"prior": prior.max(axis=1)[0].mean()})
        pseudolabels = self.prototypes.pseudo_labels(
            feat, prior, confidence_monitor=self.intensity_ma
        )
        # calculation also soft predictions to record confidence
        soft_predictions = self.prototypes.pseudo_labels(feat, prior, soft=True)
        batch_size, channels, width, height = pred_trg_ema_main["out"].size()
        self.record_ece(
            "pure prototypes",
            soft_predictions.reshape(batch_size, width, height, channels).permute(
                0, 3, 1, 2
            ),
            batch["label"],
        )
        self.intensity_ma.add(
            {"pseudolabel confidence": soft_predictions.max(axis=1)[0].mean()}
        )
        return {
            "ema_model": pred_trg_ema_main,
            "pseudolabels": pseudolabels,
            "soft_predictions": soft_predictions,
        }

    def pseudolabel_loss(self, batch):
        """From batch calculate the loss to train the model with pseudolabels"""
        soft_labels = (
            False
            if self.cfg_spec.SOFT_LABELS is None or self.cfg_spec.SOFT_LABELS == {}
            else self.cfg_spec.SOFT_LABELS
        )
        _, pred_trg_main = self.model(
            batch["image"].to(self.device)
        )  # Get model prediction
        pred_model = pred_trg_main["out"].detach().cpu()
        if self.cfg_spec.PREDICTION_SAVE != {}:
            self.save_prediction(pred_model)
        self.intensity_ma.add(
            {"model": pred_model.softmax(axis=1).max(axis=1)[0].mean()}
        )
        proto_pred = self.prototype_predictions(batch)
        self.prototypes.ma(
            proto_pred["ema_model"]["feat"], proto_pred["ema_model"]["out"]
        )
        batch_size, channels, w, h = pred_trg_main["out"].size()
        # calculate crossentropy loss
        (
            ce_loss,
            rce_loss,
            sym_loss,
            regularization_loss,
            js_divergance_loss,
            model_regularization,
        ) = (0,) * 6
        if soft_labels:
            predictions = (
                proto_pred["soft_predictions"]
                .reshape(batch_size, w, h, channels)
                .permute(0, 3, 1, 2)
            )
        else:
            predictions = proto_pred["pseudolabels"].reshape(batch_size, w, h)
        if self.cfg_spec.RCE_ALPHA > 0:
            ce_loss = loss_calc(
                pred_trg_main["out"], predictions, self.device, soft=soft_labels
            )
            sym_loss += self.cfg_spec.RCE_ALPHA * ce_loss
        if self.cfg_spec.RCE_BETA > 0:
            rce_loss = rce(
                pred_trg_main["out"], predictions, self.device, soft=soft_labels
            )
            sym_loss += self.cfg_spec.RCE_BETA * rce_loss  # sum both of the values
        total_loss = sym_loss
        if self.cfg_spec.REGULARIZER_WEIGHT > 0:  # calculating regularization loss
            regularization_loss = regular_loss(
                self.cfg_spec.REGULARIZER, pred_trg_main["out"]
            )
            total_loss += self.cfg_spec.REGULARIZER_WEIGHT * regularization_loss
        if self.cfg_spec.JS_D > 0:  # JS Divergance for noise robustness
            js_divergance_loss = js_divergance(
                pred_trg_main["out"], predictions, self.device
            )
            total_loss += self.cfg_spec.JS_D * js_divergance_loss
        if self.model_regularization is not None:
            model_regularization = self.model_regularization(self.model.parameters())
            total_loss += model_regularization

        # Validation, Recording
        current_losses = {
            "ce_loss": ce_loss,
            "pseudolabel_pixel_num": (
                (proto_pred["pseudolabels"] >= 0) * (proto_pred["pseudolabels"] != 255)
            )
            .float()
            .sum(),
            "output & prototype agreement": (
                proto_pred["pseudolabels"].reshape(batch_size, w, h)
                == pred_trg_main["out"].argmax(axis=1)
            )
            .float()
            .mean(),
            "mean_prototype_intensity_values": (self.prototypes.prototypes**2).mean(),
            "rce_loss": rce_loss,
            "sym_loss": sym_loss,
            "regularization_loss": regularization_loss,
            "JS Divergance loss": js_divergance_loss,
            "Total target loss": total_loss,
            "model regularization": model_regularization,
        }
        for name, value in self.intensity_ma.avg().items():
            current_losses[f"{name} confidence ma"] = value
        for name, value in self.intensity_ma.exp().items():
            current_losses[f"{name} exp confidence ma"] = value
        current_losses["dev avg prior static"] = self.intensity_ma.dev_avg(
            "prior static"
        )
        batch["stored_predictions"] = (
            proto_pred["soft_predictions"]
            .reshape(batch_size, w, h, channels)
            .permute(0, 3, 1, 2)
        )
        return current_losses

    def evaluate(self, validation_loader):
        """Evaluates the current model"""

        def proto_func(batch):
            proto_pred = self.prototype_predictions(batch)
            batch_size, channels, width, height = proto_pred["ema_model"]["out"].size()
            return (
                proto_pred["soft_predictions"]
                .reshape(batch_size, width, height, channels)
                .permute(0, 3, 1, 2)
            )

        if (
            isinstance(self.cfg_spec.SKIP_PROTO_EVAL, bool) != {}
            and self.cfg_spec.SKIP_PROTO_EVAL
        ):
            evaluation = super().evaluate(validation_loader)
        else:
            evaluation = super().evaluate(validation_loader, {"proto": proto_func})
        self.register_ece()
        return evaluation

    def evaluate_update_dynamic(self):
        if self.cfg_spec.AUTO_DYNAMIC != {} and self.cfg_spec.AUTO_DYNAMIC:
            self.dynamic_update_counter += 1
            if self.dynamic_update_counter > 500:
                x = self.intensity_ma.dev_avg("prior static")
                if isinstance(x, torch.Tensor):
                    x = x.item()
                if np.abs(x) > self.cfg_spec.DEV_THRESH:
                    self.update_dynamic()
                    self.dynamic_update_counter = 0

    def update_ema(self):
        for param_q, param_k in zip(
            self.model.parameters(), self.ema_model.parameters()
        ):
            param_k.data = (
                param_k.data.clone() * self.cfg_spec.EMA_UPDATE
                + param_q.data.clone() * (1.0 - self.cfg_spec.EMA_UPDATE)
            )
        for buffer_q, buffer_k in zip(self.model.buffers(), self.ema_model.buffers()):
            buffer_k.data = buffer_q.data.clone()

    def step(self, batches_source, batch_target):
        """
        Performs one learning step:
        Args:
            batches_source: list of the source samples
            batch_target: target sample, where network performs adaptation to
        """
        # buffer(source) training
        loss_seg_src_main = {}
        if self.cfg_spec.BN_POLICY == "freeze":
            switch_batch_statistics(self.model, False)
        elif self.cfg_spec.BN_POLICY == "double":
            self.bn.exchange()
        for batch_source in batches_source:
            if self.cfg.TRAINING.REPLAY_BUFFER > 0:
                # train on source
                loss_seg_src_main = self.supervised_loss(batch_source)
                loss_seg_src_main["buff_loss"].backward()
                # print([ x.split('/')[-1] for x in batch_source['image_path']], loss_seg_src_main['buff_loss'].item())
        if self.cfg_spec.BN_POLICY == "freeze":
            switch_batch_statistics(self.model, True)
        elif self.cfg_spec.BN_POLICY == "double":
            self.bn.exchange()
        # self.bn.exchange()
        # target data
        pseudolabel_losses = self.pseudolabel_loss(batch_target)
        pseudolabel_losses["Total target loss"].backward()
        # print([ x.split('/')[-1] for x in batch_target['image_path']], pseudolabel_losses['Total target loss'].item())
        pseudolabel_losses["encoder_lr"] = self.optimizer.param_groups[0]["lr"]
        pseudolabel_losses.update(loss_seg_src_main)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return pseudolabel_losses
        # return loss_seg_src_main

    def buffer_update(self, batch_target, probability, trainloader):
        total_buffer_updates = 0
        if probability > 0:
            random_nums = np.random.rand(len(batch_target["stored_predictions"]))
            indexes_to_add = np.where(random_nums < probability)[0]
            for index in indexes_to_add:
                batch_target["stored_predictions"] = self.interp(
                    batch_target["stored_predictions"]
                ).argmax(axis=1)
                trainloader.add_from_batch(batch_target, index)
                total_buffer_updates += 1
        return total_buffer_updates

    def train(self, trainloader, targetloader, validation_loaders):
        if self.cfg_spec.AUTO_DYNAMIC == {} or self.cfg_spec.AUTO_DYNAMIC == False:
            self.update_dynamic()
        if not self.cfg_spec.SKIP_CALC:
            # initializing prototypes
            if not self.skip_proto:
                print("Computing Prototypes")
                switch_batch_statistics(self.model, False)
                if self.cfg_spec.STARTING_PROTO == "target":
                    self.calculate_prototypes(targetloader)
                elif self.cfg_spec.STARTING_PROTO == "source":
                    self.calculate_prototypes(trainloader)
                switch_batch_statistics(self.model, True)
                self.skip_proto = True
            # evaluation
            print("Model evaluation")
            wandb.log(self.evaluate_all(validation_loaders))
        steps = self.cfg_spec.EPOCHS * len(targetloader)
        trainloader_iter = iter(trainloader)
        targetloader_iter = iter(targetloader)
        self.optimizer.zero_grad()
        update_prob = self.probability_per_step / steps
        samples_every = self.cfg.OTHERS.GENERATE_SAMPLES_EVERY
        for i_iter in tqdm(range(steps)):
            self.adjust_learning_rate(i_iter, steps)
            source_samples = []
            for _ in range(self.cfg_spec.SOURCE_REPEAT):
                try:
                    source_sample = next(trainloader_iter)
                except StopIteration:
                    print("------------------------")
                    # torch.cuda.synchronize(device=self.device)
                    trainloader_iter = iter(trainloader)
                    source_sample = next(trainloader_iter)
                source_samples.append(source_sample)
            try:
                target_sample = next(targetloader_iter)
            except StopIteration:
                targetloader_iter = iter(targetloader)
                target_sample = next(targetloader_iter)
            self.speed_monitor.reset_timer()
            log = self.step(source_samples, target_sample)
            self.speed_monitor.add('step_loop_time')
            log.update(self.speed_monitor.avg())
            self.evaluate_update_dynamic()
            self.update_ema()
            log["Total buffer updates"] = self.buffer_update(
                target_sample, update_prob, trainloader
            )
            if (i_iter + 1) % len(targetloader) == 0:
                print("Model evaluation")
                evaluation_log = self.evaluate_all(validation_loaders)
                log.update(evaluation_log)
                if (i_iter + 1) % len(targetloader) % samples_every == 0:
                    log.update(self.test_on_samples(validation_loaders))
                self.save_model()
            wandb.log(log)
        self.save_model()

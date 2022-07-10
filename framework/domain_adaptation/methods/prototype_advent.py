import torch
import wandb
from tqdm import tqdm
from framework.utils.loss import rce, js_divergance
from framework.utils.func import loss_calc
from framework.domain_adaptation.methods.prototypes import regular_loss
from framework.domain_adaptation.methods.prototypes_hswitch import (
    hswitch_proDA,
    switch_batch_statistics,
)
from framework.domain_adaptation.methods.advent_da import advent


class adv_proDA:
    def __init__(self, model, cfg, cfg_spec) -> None:
        self.proto_model = hswitch_proDA(model, cfg, cfg_spec)
        self.advent = advent(model, cfg, cfg_spec)

    def update_cfg_spec(self, cfg_spec):
        self.proto_model.update_cfg_spec(cfg_spec)
        self.advent.update_cfg_spec(cfg_spec)

    def step(self, batch_source, batch_target):
        soft_labels = (
            False
            if self.proto_model.cfg_spec.SOFT_LABELS is None
            or self.proto_model.cfg_spec.SOFT_LABELS == {}
            else self.proto_model.cfg_spec.SOFT_LABELS
        )

        # supervised loss
        self.advent.bn.exchange()
        self.advent.discriminator_grad(False)
        pred_src_aux, pred_src_main = self.advent.model(
            batch_source["image"].to(self.advent.device)
        )
        if pred_src_aux is not None:
            pred_src_aux = self.advent.interp(pred_src_aux["out"])
        pred_src_main = self.advent.interp(pred_src_main["out"])
        loss_seg_src = self.advent.supervised_loss(
            pred_src_aux, pred_src_main, batch_source["label"]
        )
        loss_seg_src.backward()
        self.advent.bn.exchange()

        pred_trg_aux, pred_trg_main = self.advent.model(
            batch_target["image"].to(self.advent.device)
        )
        # prototypical pseudolabeling
        proto_pred = self.proto_model.prototype_predictions(batch_target)
        self.proto_model.prototypes.ma(
            proto_pred["ema_model"]["feat"], proto_pred["ema_model"]["out"]
        )
        batch_size, channels, w, h = pred_trg_main["out"].size()
        ce_loss, rce_loss, sym_loss, regularization_loss, js_divergance_loss = (0,) * 5
        if soft_labels:
            predictions = (
                proto_pred["soft_predictions"]
                .reshape(batch_size, w, h, channels)
                .permute(0, 3, 1, 2)
            )
        else:
            predictions = proto_pred["pseudolabels"].reshape(batch_size, w, h)
        if self.proto_model.cfg_spec.RCE_ALPHA > 0:
            ce_loss = loss_calc(
                pred_trg_main["out"],
                predictions,
                self.proto_model.device,
                soft=soft_labels,
            )
            sym_loss += self.proto_model.cfg_spec.RCE_ALPHA * ce_loss
        if self.proto_model.cfg_spec.RCE_BETA > 0:
            rce_loss = rce(
                pred_trg_main["out"],
                predictions,
                self.proto_model.device,
                soft=soft_labels,
            )
            sym_loss += (
                self.proto_model.cfg_spec.RCE_BETA * rce_loss
            )  # sum both of the values
        total_loss = sym_loss
        if (
            self.proto_model.cfg_spec.REGULARIZER_WEIGHT > 0
        ):  # calculating regularization loss
            regularization_loss = regular_loss(
                self.proto_model.cfg_spec.REGULARIZER, pred_trg_main["out"]
            )
            total_loss += (
                self.proto_model.cfg_spec.REGULARIZER_WEIGHT * regularization_loss
            )
        if self.proto_model.cfg_spec.JS_D > 0:  # JS Divergance for noise robustness
            js_divergance_loss = js_divergance(
                pred_trg_main["out"], predictions, self.proto_model.device
            )
            total_loss += self.proto_model.cfg_spec.JS_D * js_divergance_loss

        # adversarial training
        if pred_trg_aux is not None:
            pred_trg_aux = self.advent.interp(pred_trg_aux["out"])
        pred_trg_main = self.advent.interp(pred_trg_main["out"])
        loss_adv = self.advent.adversarial_loss(pred_trg_aux, pred_trg_main)
        (total_loss + loss_adv).backward()

        # train discriminators
        self.advent.discriminator_grad(True)
        # train with source data
        loss_d_source, loss_d_target = self.advent.discriminator_loss(
            pred_src_aux, pred_src_main, pred_trg_aux, pred_trg_main
        )
        # train with target
        d_loss = loss_d_source + loss_d_target
        d_loss.backward()

        self.advent.optimizer.step()
        self.advent.optimizer.zero_grad()
        if pred_trg_aux is not None:
            self.advent.optimizer_d_aux.step()
            self.advent.optimizer_d_aux.zero_grad()
        self.advent.optimizer_d_main.step()
        self.advent.optimizer_d_main.zero_grad()

        current_losses = {
            "Discriminator loss": d_loss,
            "Segmentation loss": loss_seg_src,
            "Adversarial loss": loss_adv,
            "pseudolabel_pixel_num": (
                (proto_pred["pseudolabels"] >= 0) * (proto_pred["pseudolabels"] != 255)
            )
            .float()
            .sum(),
            "mean_prototype_intensity_values": (
                self.proto_model.prototypes.prototypes**2
            ).mean(),
            "rce_loss": rce_loss,
            "sym_loss": sym_loss,
            "regularization_loss": regularization_loss,
            "JS Divergance loss": js_divergance_loss,
            "Total target loss": total_loss,
        }

        for name, value in self.proto_model.intensity_ma.avg().items():
            current_losses[f"{name} confidence ma"] = value
        current_losses["dev avg prior static"] = self.proto_model.intensity_ma.dev_avg(
            "prior static"
        )
        batch_target["stored_predictions"] = (
            proto_pred["soft_predictions"]
            .reshape(batch_size, w, h, channels)
            .permute(0, 3, 1, 2)
        )
        return current_losses

    def train(self, trainloader, targetloader, validation_loaders):
        self.proto_model.update_dynamic()
        if not self.proto_model.cfg_spec.SKIP_CALC:
            if not self.proto_model.skip_proto:
                print("Computing Prototypes")
                switch_batch_statistics(self.proto_model.model, False)
                if self.proto_model.cfg_spec.STARTING_PROTO == "target":
                    self.proto_model.calculate_prototypes(targetloader)
                elif self.proto_model.cfg_spec.STARTING_PROTO == "source":
                    self.proto_model.calculate_prototypes(trainloader)
                switch_batch_statistics(self.proto_model.model, True)
                self.proto_model.skip_proto = True
            # evaluation
            print("Model evaluation")
            wandb.log(self.proto_model.evaluate_all(validation_loaders))
        steps = self.proto_model.cfg_spec.EPOCHS * len(targetloader)
        trainloader_iter = iter(trainloader)
        targetloader_iter = iter(targetloader)
        self.advent.optimizer.zero_grad()
        self.advent.optimizer_d_main.zero_grad()
        self.advent.optimizer_d_aux.zero_grad()
        samples_every = self.advent.cfg.OTHERS.GENERATE_SAMPLES_EVERY
        for i_iter in tqdm(range(steps)):
            self.advent.adjust_learning_rate(i_iter, steps)
            try:
                source_sample = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                source_sample = next(trainloader_iter)
            try:
                target_sample = next(targetloader_iter)
            except StopIteration:
                targetloader_iter = iter(targetloader)
                target_sample = next(targetloader_iter)
            log = self.step(source_sample, target_sample)
            self.proto_model.update_ema()
            if (i_iter + 1) % len(targetloader) == 0:
                print("Model evaluation")
                evaluation_log = self.proto_model.evaluate_all(validation_loaders)
                log.update(evaluation_log)
                if (i_iter + 1) % len(targetloader) % samples_every == 0:
                    log.update(self.proto_model.test_on_samples(validation_loaders))
            wandb.log(log)
        self.advent.save_model()
        self.proto_model.save_model()

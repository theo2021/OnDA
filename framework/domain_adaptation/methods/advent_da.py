import os
from copy import deepcopy
from statistics import mode
import torch
import numpy as np
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
from framework.utils.monitoring import Monitor, PytorchSpeedMeasure
from framework.domain_adaptation.evaluate import segment_sample
from framework.domain_adaptation.methods.prototype_handler import prototype_handler
from framework.model.discriminator import get_fc_discriminator
from framework.domain_adaptation.methods.adaptation_model import (
    da_model,
    switch_batch_statistics,
)


def switch_batch_statistics(model, setting):
    """Frezes and defreezes all batch norm update from a model"""
    assert isinstance(
        setting, bool
    ), f"setting value should be a boolean, given: {setting}"
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = setting


class advent(da_model):
    source_label = 0
    target_label = 1

    def __init__(self, model, cfg, cfg_spec) -> None:
        # Creating the models
        super(advent, self).__init__(model, cfg, cfg_spec)
        num_classes = cfg.NUM_CLASSES
        self.d_aux = get_fc_discriminator(num_classes=num_classes)
        self.d_aux.train()
        self.d_aux.to(self.device)
        # seg maps, i.e. output, level
        self.d_main = get_fc_discriminator(num_classes=num_classes)
        self.d_main.train()
        self.d_main.to(self.device)
        self.optimizer_d_aux = optim.Adam(
            self.d_aux.parameters(), lr=cfg_spec.LEARNING_RATE_D, betas=(0.9, 0.99)
        )
        self.optimizer_d_main = optim.Adam(
            self.d_main.parameters(), lr=cfg_spec.LEARNING_RATE_D, betas=(0.9, 0.99)
        )

    def save_model(self):
        super().save_model(
            model_dict={
                "model": self.model,
                "d_main": self.d_main,
                "d_aux": self.d_aux,
            },
            prefix="current",
        )

    def models_eval(self):
        self.model.eval()

    def models_default_config(self):
        self.model.train()

    def discriminator_grad(self, option):
        for param in self.d_aux.parameters():
            param.requires_grad = option
        for param in self.d_main.parameters():
            param.requires_grad = option

    def supervised_loss(self, pred_src_aux, pred_src_main, label):
        loss_seg_src_aux = 0
        if pred_src_aux is not None:
            loss_seg_src_aux = loss_calc(pred_src_aux, label, self.device)
        loss_seg_src_main = loss_calc(pred_src_main, label, self.device)
        return (
            self.cfg_spec.LAMBDA_SEG_MAIN * loss_seg_src_main
            + self.cfg_spec.LAMBDA_SEG_AUX * loss_seg_src_aux
        )

    def adversarial_loss(self, pred_trg_aux, pred_trg_main):
        loss_adv_trg_aux = 0
        if pred_trg_aux is not None:
            d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, self.source_label)
        d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, self.source_label)
        return (
            self.cfg_spec.LAMBDA_ADV_MAIN * loss_adv_trg_main
            + self.cfg_spec.LAMBDA_ADV_AUX * loss_adv_trg_aux
        )

    def discriminator_loss(
        self, pred_src_aux, pred_src_main, pred_trg_aux, pred_trg_main
    ):
        loss_d_src_aux = 0
        if pred_src_aux is not None:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_src_aux = bce_loss(d_out_aux, self.source_label) / 2
        pred_src_main = pred_src_main.detach()
        d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_src_main = bce_loss(d_out_main, self.source_label) / 2
        loss_d_source = loss_d_src_main + loss_d_src_aux
        loss_d_trg_aux = 0
        if pred_trg_aux is not None:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_trg_aux = bce_loss(d_out_aux, self.target_label) / 2
        pred_trg_main = pred_trg_main.detach()
        d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_trg_main = bce_loss(d_out_main, self.target_label) / 2
        loss_d_target = loss_d_trg_main + loss_d_trg_aux

        return loss_d_source, loss_d_target

    def step(self, batch_source, batch_target):
        """
        Performs one learning step:
        Args:
            batches_source: list of the source samples
            batch_target: target sample, where network performs adaptation to
        """
        self.discriminator_grad(False)
        switch_batch_statistics(self.model, False)
        # task training
        pred_src_aux, pred_src_main = self.model(batch_source["image"].to(self.device))
        if pred_src_aux is not None:
            pred_src_aux = self.interp(pred_src_aux["out"])
        pred_src_main = self.interp(pred_src_main["out"])
        loss_seg_src = self.supervised_loss(
            pred_src_aux, pred_src_main, batch_source["label"]
        )
        loss_seg_src.backward()
        switch_batch_statistics(self.model, True)
        # adversarial training
        pred_trg_aux, pred_trg_main = self.model(batch_target["image"].to(self.device))
        if pred_trg_aux is not None:
            pred_trg_aux = self.interp(pred_trg_aux["out"])
        pred_trg_main = self.interp(pred_trg_main["out"])
        loss_adv = self.adversarial_loss(pred_trg_aux, pred_trg_main)
        loss_adv.backward()

        # train discriminators
        self.discriminator_grad(True)
        # train with source data

        loss_d_source, loss_d_target = self.discriminator_loss(
            pred_src_aux, pred_src_main, pred_trg_aux, pred_trg_main
        )

        # train with target
        d_loss = loss_d_source + loss_d_target
        d_loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        if pred_trg_aux is not None:
            self.optimizer_d_aux.step()
            self.optimizer_d_aux.zero_grad()
        self.optimizer_d_main.step()
        self.optimizer_d_main.zero_grad()

        return {
            "Discriminator loss": d_loss,
            "Segmentation loss": loss_seg_src,
            "Adversarial loss": loss_adv,
        }

    def train(self, trainloader, targetloader, validation_loaders):
        if not self.cfg_spec.SKIP_CALC:
            wandb.log(self.evaluate_all(validation_loaders))
        steps = self.cfg_spec.EPOCHS * len(targetloader)
        trainloader_iter = iter(trainloader)
        targetloader_iter = iter(targetloader)
        self.optimizer.zero_grad()
        self.optimizer_d_main.zero_grad()
        self.optimizer_d_aux.zero_grad()
        samples_every = self.cfg.OTHERS.GENERATE_SAMPLES_EVERY
        for i_iter in tqdm(range(steps)):
            self.adjust_learning_rate(i_iter, steps)
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
            if (i_iter + 1) % len(targetloader) == 0:
                print("Model evaluation")
                evaluation_log = self.evaluate_all(validation_loaders)
                log.update(evaluation_log)
                if (i_iter + 1) % len(targetloader) % samples_every == 0:
                    log.update(self.test_on_samples(validation_loaders))
                self.save_model()
            wandb.log(log)
        self.save_model()

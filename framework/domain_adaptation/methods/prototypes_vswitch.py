import torch
from framework.domain_adaptation.methods.prototypes import online_proDA


class model_select:
    static = 0
    dynamic = 1

    def __init__(self, start=0, threshold_c=0.00028) -> None:
        self.current = start
        self.freeze = False
        self.threshold = threshold_c

    def eval(self):
        self.freeze = True

    def train(self):
        self.freeze = False

    def evaluate(self, dev_value):
        if not self.freeze:
            if dev_value > self.threshold:
                self.current = self.static
            elif dev_value < -self.threshold:
                self.current = self.dynamic


class vswitch_proDA(online_proDA):
    def __init__(self, model, cfg, cfg_spec) -> None:
        # Creating the models
        super(vswitch_proDA, self).__init__(model, cfg, cfg_spec)
        self.model_select = model_select(
            model_select.static, cfg_spec.SWITCH_PRIOR_THRESH
        )

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
            prev = self.model_select.current
            self.model_select.evaluate(self.intensity_ma.dev_avg("prior static"))
            # if prev != self.model_select.current:
            #     self.update_dynamic()
            # Dynamic Model
            if (
                self.model_select.current == model_select.dynamic
                and self.cfg_spec.DYNAMIC_LAMBDA > 0
            ):
                _, pred_trg_dynamic_main = self.dynamic_model(batch_image)
                prior_dynamic = pred_trg_dynamic_main["out"].softmax(axis=1)
                self.intensity_ma.add(
                    {"prior dynamic": prior_dynamic.max(axis=1)[0].mean()}
                )
                prior = self.cfg_spec.DYNAMIC_LAMBDA * prior_dynamic
                self.record_ece("dynamic", prior_dynamic, batch["label"])
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

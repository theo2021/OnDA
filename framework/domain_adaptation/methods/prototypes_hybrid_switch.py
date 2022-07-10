import torch
from framework.domain_adaptation.methods.prototypes import online_proDA


class model_select:
    static = 0
    dynamic = 1

    def __init__(self, start=0, gray_area=(0.84, 0.88), dev_threshold=0.0002) -> None:
        self.current = start
        self.freeze = False
        self.current_dev = start
        self.gray_area = gray_area
        self.dev_threshold = dev_threshold

    def eval(self):
        self.freeze = True

    def train(self):
        self.freeze = False

    def evaluate(self, confidence, dev_value):
        if not self.freeze:
            if dev_value > self.dev_threshold:
                self.current_dev = self.static
            elif dev_value < -self.dev_threshold:
                self.current_dev = self.dynamic

            if confidence < self.gray_area[0]:
                self.current = self.dynamic
            elif confidence > self.gray_area[1]:
                self.current = self.static
            else:
                self.current = self.current_dev


class hybrid_proDA(online_proDA):
    def __init__(self, model, cfg, cfg_spec) -> None:
        # Creating the models
        self.model_select = model_select(
            model_select.static, cfg_spec.GRAY_AREA, cfg_spec.DEV_THRESH
        )
        super().__init__(model, cfg, cfg_spec)

    def prototype_predictions(self, batch):
        """From batch calculate Target prototype predictions"""
        with torch.no_grad():
            batch_image = batch["image"].to(self.device)
            if "label" not in batch:  # if label is not provided to this to avoid errors
                batch["label"] = 0
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
                    {"prior static": prior_static.max(axis=1)[0].mean().item()}
                )
                prior += self.cfg_spec.STATIC_LAMBDA * prior_static
                self.record_ece("static", prior_static, batch["label"])
            if self.cfg_spec.EXP_PR_STATIC != {} and self.cfg_spec.EXP_PR_STATIC:
                static_conf = self.intensity_ma.exp("prior static")
            else:
                static_conf = self.intensity_ma.avg("prior static")
            self.model_select.evaluate(
                static_conf, self.intensity_ma.dev_avg("prior static")
            )
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
                self.record_ece("dynamic", prior_dynamic, batch["label"])
                prior = self.cfg_spec.DYNAMIC_LAMBDA * prior_dynamic
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

    def models_eval(self):
        self.model_select.eval()
        return super().models_eval()

    def models_default_config(self):
        self.model_select.train()
        return super().models_default_config()

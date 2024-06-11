from ..model.deeplabv2 import get_deeplab_v2
from ..model.deeplabv2_proda import Deeplab
from ..model.SegFormer import SegFormerMitB1Model
import torch
import types

MODEL_NAMES = [
    "DeepLabv2-Resnet50",
    "DeepLabv2-Resnet101",
    "DeepLabv2-Resnet101-ProDA",
    "DeepLabv2-Resnet50-GN",
    "SegFormerMitB1Model"
]


def get_model(cfg, n_classes):
    assert cfg.MODEL.NAME in MODEL_NAMES, f"cfg.MODEL.NAME should be in {MODEL_NAMES}"
    if cfg.MODEL.NAME == "DeepLabv2-Resnet50":
        resnet50 = [3, 4, 6, 3]
        model = get_deeplab_v2(
            num_classes=n_classes,
            layers=resnet50,
            multi_level=True,
            classifier=cfg.MODEL.CLASSIFIER,
        )
    elif cfg.MODEL.NAME == "DeepLabv2-Resnet101":
        model = get_deeplab_v2(
            num_classes=n_classes, multi_level=True, classifier=cfg.MODEL.CLASSIFIER
        )
    elif cfg.MODEL.NAME == "DeepLabv2-Resnet101-ProDA":
        cfg.MODEL.MULTI_LEVEL = False
        model = Deeplab(torch.nn.BatchNorm2d, num_classes=n_classes)
    elif cfg.MODEL.NAME == "DeepLabv2-Resnet50-GN":
        resnet50 = [3, 4, 6, 3]
        norm_module = lambda *k, **x: torch.nn.GroupNorm(32, *k, **x)
        model = get_deeplab_v2(
            num_classes=n_classes,
            layers=resnet50,
            multi_level=True,
            classifier=cfg.MODEL.CLASSIFIER,
            norm_module=norm_module,
        )
    elif cfg.MODEL.NAME == "SegFormerMitB1Model":
        model = SegFormerMitB1Model(num_classes=n_classes)
    if cfg.MODEL.LOAD is not None:
        if not cfg.MODEL.LOAD == "None":
            saved_state_dict = torch.load(cfg.MODEL.LOAD)
            if isinstance(saved_state_dict, types.MethodType):
                saved_state_dict = saved_state_dict()
            if "imagenet" in cfg.MODEL.LOAD.lower():
                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    i_parts = i.split(".")
                    ind = 0
                    if "Scale" == i_parts[0] or "module" == i_parts[0]:
                        ind = 1
                    if not (i_parts[ind] == "layer5" or i_parts[ind] == "fc"):
                        new_params[".".join(i_parts[ind:])] = saved_state_dict[i]
                model.load_state_dict(new_params)
            elif 'mitb1' in cfg.MODEL.LOAD.lower():
                saved_state_dict = saved_state_dict if 'state_dict' not in saved_state_dict.keys() else saved_state_dict['state_dict']
                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    i_parts = i.split(".")
                    if "decode_head" == i_parts[0] and ("linear_fuse_m1" == i_parts[1] or "linear_pred_m1" == i_parts[1]):
                        continue
                    new_params[i] = saved_state_dict[i]
                model.load_state_dict(new_params)
            else:
                model.load_state_dict(saved_state_dict)
    model.multi_level = cfg.MODEL.MULTI_LEVEL
    model.to(cfg.OTHERS.DEVICE)
    return model

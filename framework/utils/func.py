import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from framework.utils.loss import cross_entropy_2d


def is_turn(iteration, every):
    return iteration % every == 0 and iteration > 0


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f"{loss_name} = {to_numpy(loss_value):.3f} ")
    full_string = " ".join(list_strings)
    tqdm.write(f"iter = {i_iter} {full_string}")


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def loss_calc(pred, label, device, soft=False):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    return cross_entropy_2d(pred, label, soft)


def lr_poly(base_lr, iter, max_iter, power):
    """Poly_LR scheduler"""
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(
    optimizer, i_iter, cfg, learning_rate, iters=None, power=None
):
    if iter is not None:
        lr = lr_poly(learning_rate, i_iter, iters, power)
    else:
        lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]["lr"] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr"] = lr * 10


def adjust_learning_rate(optimizer, i_iter, cfg):
    """adject learning rate for main segnet"""
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)


def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)


def prob_2_entropy(prob):
    """convert probabilistic prediction maps to weighted self-information maps"""
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (
        hist.sum(1) + hist.sum(0) - np.diag(hist) + np.finfo(float).eps
    )


class color_mapper:
    """
    An object that performs a fast class assignment to the pixels,
    Huge thanks to Daniel: https://stackoverflow.com/questions/33196130/replacing-rgb-values-in-numpy-array-by-integer-is-extremely-slow/33196320#33196320
    """

    def __init__(self, map_dict):
        if (
            type(next(iter(map_dict.keys()))) == tuple
            or type(next(iter(map_dict.keys()))) == list
        ):
            self.rgb = True
            self.color_map = np.ndarray(shape=(256 * 256 * 256), dtype="int32")
            self.color_map[:] = 0
            for rgb, idx in map_dict.items():
                rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
                self.color_map[rgb] = idx
        else:
            self.rgb = False
            self.color_map = np.zeros(len(map_dict.keys()), dtype=np.int)
            for source, target in map_dict.items():
                self.color_map[source] = target

    def __call__(self, image):
        image = np.array(image, dtype="int32")
        if self.rgb:
            image = image.dot(np.array([65536, 256, 1], dtype="int32"))
        return self.color_map[image]

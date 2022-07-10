import numpy as np
import torch
from torch.backends import cudnn
import torch.nn.functional as F
from torch.autograd import Variable

cudnn.benchmark = False
cudnn.enabled = True
cudnn.deterministic = True


def CXE(predicted, target):
    return -(target * torch.log(predicted + 1e-6)).sum(dim=1).mean()


def cross_entropy_2d(predict, target, soft=False):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3 or target.dim() == 4
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(-2) == target.size(
        -2
    ), f"{predict.size(-2)} vs {target.size(-1)}"
    assert predict.size(-1) == target.size(
        -1
    ), f"{predict.size(-1)} vs {target.size(-1)}"
    if soft:
        loss = CXE(predict.contiguous(), target)
        return loss
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    # the balancing_constant is used to not add more weight to the infered pseudolabels because some were weak
    # balancing_constant = target_mask.float().sum()/np.product([*target_mask.shape])
    loss = F.cross_entropy(predict, target, size_average=True)  # *balancing_constant
    return loss


def entropy_loss(v):
    """
    Entropy loss for probabilistic prediction vectors
    input: batch_size x channels x h x w
    output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


# From ProDA


def js_divergance(pred, labels, device):
    """
    Jensen–Shannon divergence
    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    pred: batch_size x channels x h x w
    labels: batch_size x 1 x h x w
    """
    pred = pred.softmax(axis=1)
    batch_size, class_numbers, h, w = pred.size()
    labels_clone = labels.clone()
    mask = (labels_clone != 255).float().to(device)
    mpred = (pred.permute(1, 0, 2, 3) * mask).permute(1, 0, 2, 3)
    labels_clone[labels_clone == 255] = class_numbers
    label_one_hot = (
        torch.nn.functional.one_hot(labels_clone, class_numbers + 1).float().to(device)
    )
    label_one_hot = torch.clamp(
        label_one_hot.permute(0, 3, 1, 2)[:, :-1, :, :], min=1e-4, max=1.0
    )
    per_pixel_entropy = (
        entropy_loss((label_one_hot + mpred) / 2)
        - (entropy_loss(label_one_hot) + entropy_loss(mpred)) / 2
    )
    return torch.sum(per_pixel_entropy) * batch_size * h * w / mask.sum()


def rce(pred, labels, device, soft=False):
    pred = pred.softmax(axis=1)
    batch_size, class_numbers, width, height = pred.shape
    labels_clone = labels.long().clone()
    # pred = torch.clamp(pred, min=1e-7, max=1.0)
    # Theoreticaly softmax output should be fine
    mask = (labels_clone != 255).float().to(device)
    if soft:
        rce_loss = -(torch.sum(pred * torch.log(labels + 1e-6), dim=1)).sum() / (
            batch_size * width * height
        )
        return rce_loss
    labels_clone[labels_clone == 255] = class_numbers
    label_one_hot = (
        torch.nn.functional.one_hot(labels_clone, class_numbers + 1).float().to(device)
    )
    label_one_hot = torch.clamp(
        label_one_hot.permute(0, 3, 1, 2)[:, :-1, :, :], min=1e-4, max=1.0
    )
    rce_loss = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (
        mask.sum() + 1e-6
    )
    # this part had mask.sum(), but this can cause instabilities on different threshold, you are empasizing too much in some things
    # np.product([*mask.shape])
    return rce_loss

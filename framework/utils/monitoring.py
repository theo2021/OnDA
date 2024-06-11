from statistics import median
import torch
from torch.nn import functional as F
import numpy as np


class Monitor(object):
    """
    Class responsible for recording measurements and returning an average of them
    Examples:

        m = monitor()
        m.add({'loss': 1, 'sep': 0.1})
        m.add({'loss': 3, 'sep': 0.5})
        print(m.avg()) # {'loss': 2, 'sep': 0.3}
    """

    def __init__(self, limit=None, exp_const=0.01, dev_func="hamming"):
        self.current_dict = {}
        self.limit = limit
        self.exp_dict = {}
        self.exp_const = exp_const
        self.freeze = False
        self.signal = np.hamming(limit - 1)
        self.signal_sum = np.sum(self.signal)
        if dev_func == "median":
            self.mean_func = median
        elif dev_func == "mean":
            self.mean_func = lambda x: np.mean(np.array(x))
        elif dev_func == "hamming":
            self.mean_func = (
                lambda x: np.sum(self.signal * np.array(x)) / self.signal_sum
            )

    def eval(self):
        self.freeze = True

    def train(self):
        self.freeze = False

    def add(self, values, reset=False):
        if self.freeze:
            return 0
        for key, val in values.items():
            if key not in self.current_dict.keys() or reset:
                self.current_dict[key] = [val]
                self.exp_dict[key] = val
            else:
                self.current_dict[key].append(val)
                if self.limit is not None and len(self.current_dict[key]) > self.limit:
                    self.current_dict[key].pop(0)
                self.exp_dict[key] = (1 - self.exp_const) * self.exp_dict[
                    key
                ] + self.exp_const * val

    def dev_avg(self, item=None):
        if item is not None:
            return self._dev_avg(item)
        output_dict = {}
        for key, vals in self.current_dict.items():
            output_dict[key] = self._dev_avg(vals)
        return output_dict

    def _dev_avg(self, item):
        if item not in self.current_dict.keys():
            return 0
        tmp = self.current_dict[item]
        if len(tmp) < self.limit:
            return 0

        end_value = self.mean_func(tmp[1:])
        start_value = self.mean_func(tmp[:-1])
        return end_value - start_value

    def exp(self, item=None):
        if item is not None:
            if item in self.exp_dict.keys():
                return self.exp_dict[item]
            else:
                return 1
        return self.exp_dict

    def avg(self, item=None):
        if item is not None:
            if item in self.current_dict.keys():
                tmp = self.current_dict[item]
                return median(tmp)
            else:
                return 1
        output_dict = {}
        for key, vals in self.current_dict.items():
            output_dict[key] = median(vals)
        return output_dict

    def reset(self):
        self.current_dict = {}


class ECE:
    def __init__(self, bins) -> None:
        self.calc_matrix = torch.zeros(bins, 3)
        self.bins = bins
        self.gap = 1.0 / bins
        # summed confidence, number of correct, total number of samples

    def record(self, prediction, label, axis):
        corect_labels = label.reshape(-1)
        prediction_confidence, prediction_label = prediction.max(axis=axis)
        prediction_confidence = prediction_confidence.reshape(-1)
        prediction_label = prediction_label.reshape(-1)

        prediction_bins = self.conf_to_bin(prediction_confidence)
        # mullmatrix = F.one_hot(prediction_bins, num_classes=self.bins).float()
        mullmatrix = torch.sparse_coo_tensor(
            [prediction_bins.tolist(), torch.arange(len(prediction_bins)).tolist()],
            torch.ones(len(prediction_bins)),
            (self.bins, len(prediction_bins)),
        )
        stacked_values = torch.vstack(
            (
                prediction_confidence,
                (prediction_label == corect_labels).float(),
                torch.ones_like(corect_labels),
            )
        )
        # self.calc_matrix += (stacked_values @ mullmatrix).T.cpu()
        self.calc_matrix += (mullmatrix.to(prediction.device) @ stacked_values.T).cpu()

    def conf_to_bin(self, prediction_confidence):
        return (prediction_confidence // self.gap).long()

    def __call__(self):
        return (
            torch.abs(self.calc_matrix[:, 0] - self.calc_matrix[:, 1]).sum()
            / self.calc_matrix[:, 2].sum()
        )


class PytorchSpeedMeasure(Monitor):
    def __init__(self, limit=10, on=True) -> None:
        if isinstance(on, dict):
            on = False
        self.switch_off = not on
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        super().__init__(limit=limit, dev_func="mean")
        self.reset_timer()

    def reset_timer(self):
        if not self.switch_off:
            self.start.record()
            self.stop.record()

    def add(self, text):
        if self.switch_off:
            return 0
        self.stop.record()
        torch.cuda.synchronize()
        time_s = self.start.elapsed_time(self.stop) / 1000
        super().add({text: time_s})
        self.start.record()


def scale_predictions(prediction_matrix, scale_from, scale_to):
    classes = prediction_matrix.shape[1]
    classes_inv = 1.0 / classes
    multi_constant = (scale_to - classes_inv) / (scale_from - classes_inv)
    return (prediction_matrix - classes_inv) * multi_constant + classes_inv

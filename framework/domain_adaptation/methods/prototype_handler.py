import torch
import pickle
import os

from torch.nn.functional import threshold


class prototype_handler:
    def __init__(
        self,
        ma_lambda=0.9999,
        tau=1,
        thresh=0,
        distance_metric="euclidean",
        confidence_regularization_threshold=1,
    ):
        self.prototypes = 0  # will be classes x features
        self.squared_mean = 0
        self.counter = 0
        self.ma_lambda = ma_lambda
        self.mask = lambda x: torch.where(x > 0, x, torch.ones_like(x))
        self.tau = tau
        self.thresh = thresh
        if distance_metric == "euclidean":
            self.distance_measure = self.distance
        elif distance_metric == "mahalanobis":
            self.distance_measure = self.mahalanobis_distance
        else:
            raise ValueError("unexpected value for attribute distance_metric")
        if isinstance(confidence_regularization_threshold, dict):
            self.confidence_regularization_threshold = 1
        else:
            self.confidence_regularization_threshold = (
                confidence_regularization_threshold
            )

    def save(self, loc="prototypes.pickle"):
        pickle.dump((self.prototypes, self.squared_mean, self.counter), open(loc, "wb"))

    def load(self, loc="prototypes.pickle"):
        if os.path.exists(loc):
            self.prototypes, self.squared_mean, self.counter = pickle.load(
                open(loc, "rb")
            )
            print("Prototypes loaded!")
            return True
        return False

    def prototype_var(self):
        var = self.squared_mean - self.prototypes**2
        return torch.sqrt(var)

    def global_var(self):
        global_squared_mean = (
            self.squared_mean.T * self.counter / self.counter.sum()
        ).T.sum(axis=0)
        global_mean = (self.prototypes.T * self.counter / self.counter.sum()).T.sum(
            axis=0
        )
        return torch.sqrt(global_squared_mean - global_mean**2)

    def append(self, feat, out):
        prototypes, sums = self.get_proto_array(feat, out)
        proto_squared, _ = self.get_proto_array(feat**2, out)

        self.counter += sums
        total_samples_mask = self.mask(self.counter)  # preventing division by 0
        if type(self.prototypes) == int:
            self.prototypes = torch.zeros_like(prototypes, dtype=torch.float)
            self.squared_mean = torch.zeros_like(prototypes, dtype=torch.float)
        diff = prototypes - (self.prototypes.T * sums).T
        diff_2 = proto_squared - (self.squared_mean.T * sums).T
        self.prototypes += (diff.T / total_samples_mask).T
        self.squared_mean += (diff_2.T / total_samples_mask).T

    def get_proto_array(self, feat, out):
        feat_t = self.transform(feat)
        out_t = self.transform(out)
        onehot = self.onehot(out_t)
        vect = onehot.T @ feat_t
        return vect, onehot.sum(axis=0)

    def onehot(self, matrix):
        onehot = torch.zeros_like(matrix).float()
        out = matrix.argmax(axis=1, keepdim=True)
        return onehot.scatter(1, out, 1)

    def ma(self, feat, out):
        prototypes, sums = self.get_proto_array(feat, out)
        proto_squared, _ = self.get_proto_array(feat**2, out)
        # only update those that have been found
        rev_mask = self.ma_lambda ** (sums > 0).float()
        sum_mask = self.mask(sums)
        self.prototypes = (self.prototypes.T * rev_mask).T + (
            (1 - rev_mask) * (prototypes.T / sum_mask)
        ).T
        self.squared_mean = (self.squared_mean.T * rev_mask).T + (
            (1 - rev_mask) * (proto_squared.T / sum_mask)
        ).T

    # def reverse_transform_match(self, matrix, match):
    #     batch, channels, h = match.size()
    #     return matrix.reshape(batch, w, h, channels).permute(0,3,1,2)

    def transform(self, matrix):
        if matrix.dim() == 2:
            return matrix
        _, channels, _, _ = matrix.size()
        return matrix.permute(0, 2, 3, 1).reshape(-1, channels)

    def mahalanobis_distance(self, feat):
        feat_t = self.transform(feat)
        distance_matrix = torch.ones(feat_t.shape[0], self.prototypes.shape[0]).to(
            feat.device
        )
        prototype_variances = self.global_var()
        for i in range(self.prototypes.shape[0]):
            dis = (feat_t - self.prototypes[i]) / prototype_variances
            dis = torch.norm(dis, 2, dim=1)
            distance_matrix[:, i] = dis
            # distribution = MultivariateNormal(self.prototypes[i], torch.diag(prototype_variances[i]))
            # distance_matrix[:, i] = torch.exp(distribution.log_prob(feat_t))
        # distance_matrix = distance_matrix / distance_matrix.sum(axis=1, keepdim=True)
        min_distances = distance_matrix.min(axis=1)[0]
        return (distance_matrix.T - min_distances).T

    def distance(self, feat):
        feat_t = self.transform(feat)
        distance_matrix = torch.ones(feat_t.shape[0], self.prototypes.shape[0]).to(
            feat.device
        )
        for i in range(self.prototypes.shape[0]):
            dis = feat_t - self.prototypes[i]
            dis = torch.norm(dis, 2, dim=1)
            distance_matrix[:, i] = dis
        # The paper subtracts the minimum
        min_distances = distance_matrix.min(axis=1)[0]
        return (distance_matrix.T - min_distances).T

    def pseudo_labels(self, feat, prior=None, soft=False, confidence_monitor=None):
        dis = self.distance_measure(feat)
        if feat.device != prior.device:
            print(
                f"vetors not in the same device, feat: {feat.device}, prior: {prior.device}"
            )
        prior = self.transform(prior) if prior is not None else 1
        prop = (-dis / self.tau).softmax(axis=1)
        if confidence_monitor is not None:
            if not confidence_monitor.freeze:
                confidence_monitor.add({"prototypes": prop.max(axis=1)[0].mean()})
                if (
                    confidence_monitor.avg("prototypes")
                    > self.confidence_regularization_threshold
                ):
                    self.tau += 0.001
                    confidence_monitor.add({"tau": self.tau})
            # pr_regularizer = 0.985 / (max(confidence_monitor.avg('prototypes'), 0.985))
            # prop = pr_regularizer*prop*0.5 + prior*(1-pr_regularizer)*0.5
        prop *= prior
        prop = prop / prop.sum(axis=1, keepdim=True)  # normalizing again
        if soft:
            return prop
        mprop, labels = prop.max(axis=1, keepdim=True)
        # discarding labels less than threshold
        labels[mprop < self.thresh] = 255
        return labels

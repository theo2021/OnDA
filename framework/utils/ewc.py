# Adopted from https://github.com/thuyngch/Overcoming-Catastrophic-Forgetting
# ------------------------------------------------------------------------------
#    Libraries
# ------------------------------------------------------------------------------
import torch
from torch import autograd
import torch.nn.functional as F

from tqdm import tqdm


# ------------------------------------------------------------------------------
# 	Compute Fisher Information Matrix (FIM)
# ------------------------------------------------------------------------------
def compute_fisher(model, X, Y):
    # Instantiate the FIM
    fishers = []
    n_samples = X.shape[0]
    for param in model.parameters():
        fishers.append(torch.zeros_like(param))

    # # Compute the FIM (get mean of gradients)
    # logits = model(X)
    # loglikelihoods = F.log_softmax(logits, dim=1)[range(n_samples), Y]
    # for i in range(n_samples):
    # 	loglikelihood = loglikelihoods[i]
    # 	loglikelihood.backward(retain_graph=True)
    # 	for idx, param in enumerate(model.parameters()):
    # 		fishers[idx] += param.grad**2
    # for idx, param in enumerate(model.parameters()):
    # 	fishers[idx] /= n_samples
    # return fishers

    # Compute the FIM (get mean of loglikelihoods)
    logits = model(X)
    loglikelihoods = F.log_softmax(logits, dim=1)[range(n_samples), Y]
    loglikelihood = loglikelihoods.mean()
    loglikelihood.backward()
    for idx, param in enumerate(model.parameters()):
        fishers[idx] += param.grad**2
    return fishers


# ------------------------------------------------------------------------------
#    Cusstom loss funtion
# ------------------------------------------------------------------------------
def ewc_loss(lamda, prev_opt_thetas, cur_thetas, fishers=1):
    loss = 0
    for i in range(len(prev_opt_thetas)):
        fisher = fishers if isinstance(fishers, int) else fishers[i]
        prev_opt_theta = prev_opt_thetas[i]
        cur_theta = cur_thetas[i]
        loss += lamda / 2 * torch.sum(fisher * (prev_opt_theta - cur_theta) ** 2)
    return loss


# ------------------------------------------------------------------------------
# 	Training process using EWC method
# ------------------------------------------------------------------------------
def train_ewc(
    model,
    train_loader,
    optimizer,
    base_loss_fn,
    lamda,
    fishers,
    prev_opt_thetas,
    epoch,
    description="",
):
    model.train()
    loss_train = 0
    pbar = tqdm(train_loader)
    pbar.set_description(description)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        cur_thetas = list(model.parameters())

        optimizer.zero_grad()
        logits = model(inputs)
        loss_crossentropy = base_loss_fn(logits, targets)
        loss_ewc = ewc_loss(lamda, prev_opt_thetas, cur_thetas, fishers)

        loss = loss_crossentropy + loss_ewc
        loss_train += loss.item()
        loss.backward()
        optimizer.step()

    loss_train /= len(train_loader)
    print("loss_train: {:.6f}".format(loss_train))
    return loss_train

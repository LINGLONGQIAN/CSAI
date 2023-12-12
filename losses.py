import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# SVAELOSS
class SVAELoss(torch.nn.Module):
    def __init__(self, args):
        super(SVAELoss, self).__init__()
        self.args = args
        self.lambda1 = torch.tensor(args.lambda1)
        self.mae = nn.L1Loss()

    def forward(self, model, x, eval_x, x_bar, m, eval_m, enc_mu, enc_logvar, dec_mu, dec_logvar, phase='train'):

        # Reconstruction Loss
        nll = -Normal(dec_mu, torch.exp(0.5 * dec_logvar)).log_prob(x).sum(1)
        mae = torch.tensor([0.0]).to(self.args.device)
        recon_loss = nll

        # Variational Encoder Loss
        KLD_enc = - self.args.beta * 0.5 * torch.sum(1 + enc_logvar - enc_mu.pow(2) - enc_logvar.exp(), 1)

        # Regularization
        l1_regularization = torch.tensor(0).float().to(self.args.device)
        for name, param in model.named_parameters():
            if 'bias' not in name:
                l1_regularization += self.lambda1 * torch.norm(param.to(self.args.device), 1)

        # Take the average
        loss = torch.mean(recon_loss) + torch.mean(KLD_enc) + l1_regularization

        return loss, torch.mean(nll).item(), torch.mean(mae).item(), torch.mean(KLD_enc).item(), l1_regularization.item()

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bcelogits = nn.BCEWithLogitsLoss()

    def forward(self, y_score, y_out, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        BCE = self.bcelogits(y_out, targets)

        y_score = y_score.view(-1)
        targets = targets.view(-1)
        intersection = (y_score * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(y_score.sum() + targets.sum() + smooth)

        Dice_BCE = BCE + dice_loss
        
        return BCE, Dice_BCE


class DiceBCE_VariationalELBO(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCE_VariationalELBO, self).__init__()
        # self.dicebceLoss = DiceBCELoss()

    def forward(self, mll, output, y):

        likelihood_samples = mll.likelihood._draw_likelihood_samples(output)
        y_out = likelihood_samples.probs.mean(0).argmax(-1)
        y_score = likelihood_samples.probs.mean(0).max(-1).values

        # Dice_BCE_loss = self.dicebceLoss(y_score, y_out, y)

        res = likelihood_samples.log_prob(y).mean(dim=0).sum(-1)

        num_batch = output.event_shape[0]
        log_likelihood = res.div(num_batch)
        kl_divergence = mll.model.variational_strategy.kl_divergence().div(mll.num_data / mll.beta)
        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in mll.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in mll.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(mll.num_data))

        if mll.combine_terms:
            return -(log_likelihood - kl_divergence + log_prior - added_loss), y_out, y_score
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss, y_out, y_score
            else:
                return log_likelihood, kl_divergence, log_prior, y_out, y_score

class VRNNLoss(nn.Module):
    def __init__(self,  lambda1, device, isreconmsk=True):
        super(VRNNLoss, self).__init__()
        self.lambda1 = torch.tensor(lambda1).to(device)
        self.device = device
        self.isreconmsk = isreconmsk

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (std_2 - std_1 +
                       (torch.exp(std_1) + (mean_1 - mean_2).pow(2)) / torch.exp(std_2) - 1)
        # kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
        #                (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        #                std_2.pow(2) - 1)

        return 0.5 * torch.sum(kld_element, 1)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def _nll_gauss(self, mean, std, x):
        ss = std.pow(2)
        norm = x.sub(mean)
        z = torch.div(norm.pow(2), ss)
        denom_log = torch.log(2 * np.pi * ss)

        # result = 0.5 * torch.sum(z + denom_log)
        result = 0.5 * torch.sum(z + denom_log, 1)
        # result = -Normal(mean, std.mul(0.5).exp_()).log_prob(x).sum(1)

        return result  # pass

    def forward(self, model, all_prior_mean, all_prior_std, all_x, all_enc_mean, all_enc_std,
                all_dec_mean, all_dec_std, msk, eval_x, eval_msk, beta=1):

        kld_loss, nll_loss, mae_loss = 0, 0, 0
        nll_loss_2 = 0

        for t in range(len(all_x)):
            kld_loss += beta * self._kld_gauss(all_enc_mean[t], all_enc_std[t], all_prior_mean[t],
                                               all_prior_std[t])

            if self.isreconmsk:

                mu = all_dec_mean[t] * msk[:, t, :]
                std = (all_dec_std[t] * msk[:, t, :]).mul(0.5).exp_()

                cov = []
                for vec in std:
                    cov.append(torch.diag(vec))
                cov = torch.stack(cov)

                nll_loss += - MultivariateNormal(mu, cov).log_prob(all_x[t] * msk[:, t, :]).sum()


                # nll_loss_2 += - Normal(all_dec_mean[t][msk[:, t, :] == 1],
                #                         all_dec_std[t][msk[:, t, :] == 1].mul(0.5).exp_()).log_prob(
                #                         all_x[t][msk[:, t, :] == 1]).sum()
                #
                #
                # nll_loss += - Normal(mu, std).log_prob(all_x[t] * msk[:, t, :]).sum()

                mae_loss += torch.abs(all_dec_mean[t][eval_msk[:, t, :] == 1] - eval_x[:, t, :][eval_msk[:, t, :] == 1]).sum()
            else:
                nll_loss += - Normal(all_dec_mean[t], all_dec_std[t].mul(0.5).exp_()).log_prob(all_x[t]).sum(1)
                mae_loss += torch.abs(all_dec_mean[t] - all_x[t]).sum(1)

        if self.isreconmsk:
            # loss = kld_loss.mean() + (mae_loss + nll_loss) / len(kld_loss)  # KL + MAE + NLL
            loss = kld_loss.mean() + (nll_loss) / len(kld_loss)  # KL + NLL
        else:
            loss = torch.mean(kld_loss + mae_loss + nll_loss)  # KL + MAE + NLL

            # nll_loss += self._nll_gauss(all_dec_mean[t], all_dec_std[t], all_x[t])  # NLL2
            # nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            # kld_loss += - beta * 0.5 * torch.sum(1 + all_enc_std[t] - all_enc_mean[t].pow(2) - all_enc_std[t].exp(), 1)
            # nll_loss += - Normal(all_dec_mean[t], all_dec_std[t].mul(0.5).exp_()).log_prob(all_x[t]).sum(1)  # NLL1
            # print('kld' + str(kld_loss) + 'nll_loss' + str(nll_loss) + 'mae_loss' + str(mae_loss))

        l1_regularization = torch.tensor(0).float().to(self.device)
        for param in model.parameters():
            l1_regularization += torch.norm(param.to(self.device), 1)

        # loss_total = loss + (self.lambda1 * l1_regularization)
        loss_total = loss
        return loss_total

class FocalLoss(nn.Module):
    def __init__(self,  lambda1, device, alpha=1, gamma=0, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.device = device
        self.lambda1 = torch.tensor(lambda1).to(device)

    def forward(self, model, inputs, targets):
        # inputs += 1e-10
        # inputs = inputs.clamp(1e-10, 1.0)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            print("BCE_loss is", BCE_loss)
            # BCE_loss = nn.BCEWithLogitsLoss()
        else:
            # if np.shape(np.where(np.isnan(inputs.cpu().detach().numpy())==True))[1]>0:
            #     print(np.shape(np.where(np.isnan(inputs.cpu().detach().numpy()) == True))[1])
            #     inputs = torch.tensor(np.nan_to_num(inputs.cpu().detach().numpy())).to(self.device)
            #     print(inputs)
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            # BCE_loss = nn.BCELoss()
        pt = torch.exp(-1*BCE_loss)
        # pt = torch.exp(-1 * BCE_loss(inputs, targets))
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        # F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss(inputs, targets)

        # Regularization
        l1_regularization = torch.tensor(0).float().to(self.device)
        for param in model.parameters():
            l1_regularization += torch.norm(param.to(self.device), 1)

        # Take the average
        # loss = torch.mean(F_loss) + (self.lambda1 * l1_regularization)
        loss = torch.mean(F_loss)

        return loss

# Asymetric Similarity Loss
class AsymSimiliarityLoss(torch.nn.Module):

    def __init__(self, beta, lambda1, device):
        super(AsymSimiliarityLoss, self).__init__()
        self.beta = beta
        self.lambda1 = lambda1
        self.device = device

    def forward(self, model, y_pred, y):
        nom = (1 + self.beta**2) * torch.sum(y_pred * y.float())
        denom = ((1 + self.beta**2) * torch.sum(y_pred * y.float())) + \
                (self.beta**2 * torch.sum((1-y_pred) * y.float())) + \
                (torch.sum(y_pred * (1 - y).float()))
        asym_sim_loss = nom / denom

        # Regularization
        l1_regularization = torch.tensor(0).float().to(self.device)
        for param in model.parameters():
            l1_regularization += torch.norm(param.to(self.device), 1)

        # Take the average
        # loss = asym_sim_loss + (self.lambda1 * l1_regularization)
        loss = asym_sim_loss

        return loss


if __name__ == '__main__':

    output = torch.randint(0,10, size = (10,)).float()
    scroe = torch.sigmoid(output)
    target = torch.randint(0,10, size = (10,)).float()
    dbce = DiceBCELoss()
    dbce_loss = dbce(scroe, output, target)
    print(dbce_loss)


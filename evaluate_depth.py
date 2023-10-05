import numpy as np
import torch.nn.functional as F
import torch


def silog_loss(preds, targets):
        
        beta = 0.15
        alpha = 1e-7
        g = torch.log(preds + alpha) - torch.log(targets + alpha)
        Dg = torch.var(g) + beta * torch.pow(torch.mean(g), 2)
        loss = 10 * torch.sqrt(Dg)
        return loss


def l1_loss(preds, targets, mask):
    """
    l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Function that takes the mean element-wise absolute value difference.

    See :class:`~torch.nn.L1Loss` for details.
    """
    
    # preds = preds[preds > max_depth] = float('-inf')

    # return F.l1_loss(preds[mask], targets[mask])
    return F.l1_loss(preds, targets)


def mse_loss(preds, targets):
    """
    l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Function that takes the mean element-wise absolute value difference.

    See :class:`~torch.nn.L1Loss` for details.
    """
    return F.mse_loss(preds, targets)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_depth_errors(gt, pred, mask=None):

    """
    Compute various depth prediction errors.

    Args:
        gt (torch.Tensor): Ground truth depth map with shape [batch_size, 1, 3, height, width].
        pred (torch.Tensor): Predicted depth map with shape [batch_size, 1, height, width].
        mask (torch.Tensor, optional): Mask to select specific elements for evaluation.
                                       If None, all elements are considered.

    Returns:
        dict: A dictionary containing various depth prediction error metrics.
            - 'silog' (float): Scale-Invariant Logarithmic Error.
            - 'abs_rel' (float): Absolute Relative Error.
            - 'log10' (float): Log 10 Error.
            - 'rms' (float): Root Mean Square Error.
            - 'sq_rel' (float): Squared Relative Error.
            - 'log_rms' (float): Log Root Mean Square Error.
            - 'd1' (float): Percentage of pixels with < 1.25 relative depth error.
            - 'd2' (float): Percentage of pixels with < 1.25^2 relative depth error.
            - 'd3' (float): Percentage of pixels with < 1.25^3 relative depth error.
    """


    dict_errors = {}

    gt[gt<=0] = 1E-5
    pred[pred<=0] = 1E-5

    # depth target [8, 1, 3, 256, 256]
    # depth pred [8, 1, 256, 256]
    # => [8, 256, 256]

    gt = gt[:,:,0,:].squeeze().cpu().numpy() # [8, 1, 3, 256, 256] -> [8, 256, 256]
    pred = pred.squeeze().cpu().numpy()

    # gt = gt[mask]
    # pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)


    dict_errors['silog'] = silog
    dict_errors['abs_rel'] = abs_rel
    dict_errors['log10'] = log10
    dict_errors['rms'] = rms
    dict_errors['sq_rel'] = sq_rel
    dict_errors['log_rms'] = log_rms
    dict_errors['d1'] = d1
    dict_errors['d2'] = d2 
    dict_errors['d3'] = d3
    

    return dict_errors
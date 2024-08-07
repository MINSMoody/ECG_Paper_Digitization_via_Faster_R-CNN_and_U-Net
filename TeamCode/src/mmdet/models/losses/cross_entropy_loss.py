# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .accuracy import accuracy
from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (torch.Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()

    # weighted element-wise losses
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_combo_loss(pred, target, label, alpha=0.25, gamma=2.0, reduction='mean', class_weight=None, dice_weight=0.5, penalty_factor=0.1, **kwargs):
    """Calculate the combined Focal, Dice, and Size Penalty loss for masks."""
    # Calculate Dice Loss
    dice_loss = mask_dice_loss(pred, target, label, reduction=reduction, class_weight=class_weight, **kwargs)

    # Calculate Focal Loss
    focal_loss = mask_focal_loss(pred, target, label, alpha=alpha, gamma=gamma, reduction=reduction, class_weight=class_weight, **kwargs)

    # Calculate Size Penalty Loss
    # size_penalty = size_penalty_loss(pred, target, label, reduction=reduction, penalty_factor=penalty_factor)

    # Combine the losses with the specified weightss
    combo_loss = dice_weight * dice_loss + (1 - dice_weight) * focal_loss 
    # + size_penalty

    return combo_loss

def x_span_size_penalty_loss(pred, target, label, reduction='mean', penalty_factor=1.0, threshold=0.5):
    """
    Calculate an adjusted size penalty loss to encourage predicted masks to span the width of the ground truth mask.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, H, W), where C is the number of classes.
        target (torch.Tensor): The ground truth mask with shape (N, H, W).
        label (torch.Tensor): The class label of the mask corresponding to each object.
        reduction (str, optional): The method used to reduce the loss. Options are "none", "mean", and "sum".
        penalty_factor (float, optional): The factor to scale the penalty loss. Defaults to 1.0.
        threshold (float, optional): Threshold to binarize the mask prediction. Defaults to 0.5.

    Returns:
        torch.Tensor: The calculated x-span size penalty loss.
    """
    num_rois = pred.size(0)
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)

    # Select class-specific predictions and apply sigmoid
    pred_slice = pred[inds, label].squeeze(1)
    pred_probs = torch.sigmoid(pred_slice)

    # Binarize the predicted and target masks
    pred_binary = (pred_probs > threshold).float()
    target_binary = (target > threshold).float()

    # Calculate the x-span of the predicted and target masks
    x_span_penalties = []
    for i in range(num_rois):
        # Calculate predicted x-span
        pred_x_indices = torch.nonzero(pred_binary[i].sum(dim=0), as_tuple=True)[0]
        if len(pred_x_indices) > 0:
            pred_x_span = pred_x_indices[-1] - pred_x_indices[0] + 1
        else:
            pred_x_span = 0

        # Calculate target x-span
        target_x_indices = torch.nonzero(target_binary[i].sum(dim=0), as_tuple=True)[0]
        if len(target_x_indices) > 0:
            target_x_span = target_x_indices[-1] - target_x_indices[0] + 1
        else:
            target_x_span = 0

        # Penalize if the predicted x-span is smaller than the target x-span
        penalty = max(0, (target_x_span - pred_x_span))
        x_span_penalties.append(penalty)

    x_span_penalty = torch.tensor(x_span_penalties, dtype=torch.float, device=pred.device)

    # Apply penalty factor
    x_span_penalty = penalty_factor * x_span_penalty

    # Reduce the penalty loss
    if reduction == 'mean':
        return x_span_penalty.mean()
    elif reduction == 'sum':
        return x_span_penalty.sum()
    else:
        return x_span_penalty

def size_penalty_loss(pred, target, label, reduction='mean', penalty_factor=0.1):
    """Calculate a size penalty loss to encourage larger masks."""
    num_rois = pred.size(0)
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    pred_probs = torch.sigmoid(pred_slice)
    pred_flat = pred_probs.view(num_rois, -1)
    target_flat = target.view(num_rois, -1)

    # Calculate the size difference
    pred_size = pred_flat.sum(dim=1)
    target_size = target_flat.sum(dim=1)

    # Penalize for being smaller
    size_penalty = (target_size - pred_size).clamp(min=0)  # Penalize only if predicted size is smaller

    if reduction == 'mean':
        return (penalty_factor * size_penalty).mean()[None]
    elif reduction == 'sum':
        return (penalty_factor * size_penalty).sum()[None]
    else:
        return penalty_factor * size_penalty

def mask_dice_loss(pred,
                   target,
                   label,
                   reduction='mean',
                   avg_factor=None,
                   class_weight=None,
                   ignore_index=None,
                   **kwargs):
    """Calculate the Dice loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'Dice loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    
    num_rois = pred.size(0)
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    pred_probs = torch.sigmoid(pred_slice)
    pred_flat = pred_probs.view(num_rois, -1)
    target_flat = target.view(num_rois, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    # Encourage larger masks by penalizing false negatives more
    dice_score = (2.0 * intersection + 1e-6) / (union + intersection - intersection + 1e-6)
    dice_loss = 1.0 - dice_score

    if class_weight is not None:
        class_weight = torch.tensor(class_weight, device=pred.device)
        dice_loss = dice_loss * class_weight[label]

    if reduction == 'mean':
        return dice_loss.mean()[None]
    elif reduction == 'sum':
        return dice_loss.sum()[None]
    else:
        return dice_loss
def mask_focal_loss(pred, target, label, alpha=0.25, gamma=2.0, reduction='mean', class_weight=None, **kwargs):
    """Calculate the Focal loss for masks."""

    # Ensure predictions are logits (apply sigmoid for probabilities)
    num_rois = pred.size(0)
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)

    # Convert logits to probabilities
    pred_probs = torch.sigmoid(pred_slice)

    # Flatten tensors and ensure contiguity
    pred_probs = pred_probs.view(num_rois, -1).contiguous()
    target_flat = target.view(num_rois, -1).contiguous()

    # Clamp values for numerical stability
    eps = 1e-8
    pred_probs = torch.clamp(pred_probs, eps, 1.0 - eps)
    target_flat = torch.clamp(target_flat, eps, 1.0 - eps)

    # Compute pt (probability of the true class)
    pt = (1 - target_flat) * (1 - pred_probs) + target_flat * pred_probs
    # BCE = F.binary_cross_entropy_with_logits(
    #     pred_slice, target, weight=class_weight, reduction='mean')[None]
    # pt = torch.exp(-BCE)  
    # Calculate Focal Loss
    focal_loss = -((1 - pt) ** gamma) * torch.log(pt)

    # Apply alpha weighting
    alpha_t = alpha * target_flat + (1 - alpha) * (1 - target_flat)
    focal_loss = alpha_t * focal_loss

    # Handle class weights if provided
    if class_weight is not None:
        class_weight = torch.tensor(class_weight, device=pred.device)
        focal_loss = focal_loss * class_weight[label].view(-1, 1)

    # Reduction
    if reduction == 'mean':
        return focal_loss.mean()[None]  # Keep the same shape as other loss outputs
    elif reduction == 'sum':
        return focal_loss.sum()[None]
    else:
        return focal_loss

def full_combo_loss(pred, target, label, alpha=0.25, gamma=2.0, reduction='mean', class_weight=None, dice_weight=0.7, penalty_factor=0.1, boundary_factor=0.2, **kwargs):
    """Calculate the combined Focal, Dice, Size Penalty, and Boundary loss for masks."""
    # Calculate Dice Loss
    dice_loss = mask_dice_loss(pred, target, label, reduction=reduction, class_weight=class_weight, **kwargs)

    # Calculate Focal Loss
    focal_loss = mask_focal_loss(pred, target, label, alpha=alpha, gamma=gamma, reduction=reduction, class_weight=class_weight, **kwargs)

    # Calculate Size Penalty Loss
    size_penalty = size_penalty_loss(pred, target, label, reduction=reduction, penalty_factor=penalty_factor)

    x_penalty = x_span_size_penalty_loss(pred, target, label, reduction=reduction, penalty_factor=penalty_factor)
    # Calculate Boundary Loss
    boundary_loss_val = boundary_loss(pred, target, label, reduction=reduction, boundary_factor=boundary_factor)

    # Combine the losses with the specified weights
    combo_loss = dice_weight * dice_loss + (1 - dice_weight) * focal_loss + size_penalty + x_penalty + boundary_loss_val

    return combo_loss


def boundary_loss(pred, target, label, reduction='mean', boundary_factor=0.1, **kwargs):
    """Calculate a boundary loss to encourage covering object contours."""
    num_rois = pred.size(0)
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    pred_probs = torch.sigmoid(pred_slice)
    pred_flat = pred_probs.view(num_rois, -1)
    target_flat = target.view(num_rois, -1)

    # Compute boundary regions
    boundary_target = torch.abs(target_flat - pred_flat)
    boundary_loss = boundary_target.sum(dim=1)

    if reduction == 'mean':
        return (boundary_factor * boundary_loss).mean()[None]
    elif reduction == 'sum':
        return (boundary_factor * boundary_loss).sum()[None]
    else:
        return boundary_factor * boundary_loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None,
                       **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]




@MODELS.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_combo_loss
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs)
        return loss_cls


@MODELS.register_module()
class CrossEntropyCustomLoss(CrossEntropyLoss):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 num_classes=-1,
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """CrossEntropyCustomLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            num_classes (int): Number of classes to classify.
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(CrossEntropyCustomLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        self.num_classes = num_classes

        assert self.num_classes != -1

        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True

    def get_cls_channels(self, num_classes):
        assert num_classes == self.num_classes
        if not self.use_sigmoid:
            return num_classes + 1
        else:
            return num_classes

    def get_activation(self, cls_score):

        fine_cls_score = cls_score[:, :self.num_classes]

        if not self.use_sigmoid:
            bg_score = cls_score[:, [-1]]
            new_score = torch.cat([fine_cls_score, bg_score], dim=-1)
            scores = F.softmax(new_score, dim=-1)
        else:
            score_classes = fine_cls_score.sigmoid()
            score_neg = 1 - score_classes.sum(dim=1, keepdim=True)
            score_neg = score_neg.clamp(min=0, max=1)
            scores = torch.cat([score_classes, score_neg], dim=1)

        return scores

    def get_accuracy(self, cls_score, labels):

        fine_cls_score = cls_score[:, :self.num_classes]

        pos_inds = labels < self.num_classes
        acc_classes = accuracy(fine_cls_score[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        return acc

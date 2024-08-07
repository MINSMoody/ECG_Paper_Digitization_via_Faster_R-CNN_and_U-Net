# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .utils import weight_reduce_loss


def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
def focal_loss(pred,
               target,
               weight=None,
               gamma=2.0,
               alpha=0.25,
               reduction='mean',
               avg_factor=None):
    """
    Calculate Focal Loss.

    Args:
        pred (torch.Tensor): The prediction tensor with shape (N, C, *).
        target (torch.Tensor): The target tensor with shape (N, *).
        weight (torch.Tensor, optional): A manual rescaling weight given to each class. Default is None.
        gamma (float, optional): Focusing parameter for Focal Loss. Default: 2.0.
        alpha (float, optional): Balance parameter for Focal Loss. Default: 0.25.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average the loss. Default: None.

    Returns:
        torch.Tensor: Computed Focal Loss.
    """
    # Flatten inputs
    pred = pred.view(-1)
    target = target.view(-1)

    # Compute the binary cross-entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

    # Apply the sigmoid function to get probabilities
    pred_sigmoid = torch.sigmoid(pred)

    # Compute the modulating factor
    p_t = (target * pred_sigmoid) + ((1 - target) * (1 - pred_sigmoid))
    modulating_factor = (1 - p_t) ** gamma

    # Compute the alpha factor
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)

    # Compute the focal loss
    focal_loss = alpha_factor * modulating_factor * bce_loss

    # Apply weights if provided
    if weight is not None:
        focal_loss *= weight

    # Reduce loss
    if reduction == 'mean':
        focal_loss = focal_loss.mean()
    elif reduction == 'sum':
        focal_loss = focal_loss.sum()

    # Apply average factor
    if avg_factor is not None:
        focal_loss /= avg_factor

    return focal_loss

def combo_loss(pred,
               target,
               weight=None,
               eps=1e-3,
               reduction='mean',
               focal_weight=1.0,
               dice_weight=1.0,
               naive_dice=False,
               gamma=2.0,
               alpha=0.25,
               avg_factor=None):
    """
    Calculate the combined Focal Loss and Dice Loss.

    Args:
        pred (torch.Tensor): The prediction tensor with shape (N, *).
        target (torch.Tensor): The target tensor with shape (N, *).
        weight (torch.Tensor, optional): A manual rescaling weight given to each prediction. Default is None.
        eps (float): Small epsilon value to avoid division by zero in Dice Loss. Default: 1e-3.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
        focal_weight (float, optional): Weight for the Focal Loss component. Default: 1.0.
        dice_weight (float, optional): Weight for the Dice Loss component. Default: 1.0.
        naive_dice (bool, optional): Use naive dice calculation. Default: False.
        gamma (float, optional): Focusing parameter for Focal Loss. Default: 2.0.
        alpha (float, optional): Balance parameter for Focal Loss. Default: 0.25.
        avg_factor (int, optional): Average factor for the combined loss. Default: None.

    Returns:
        torch.Tensor: Computed combined loss.
    """
    # Calculate Dice Loss
    input_flat = pred.flatten(1)
    target_flat = target.flatten(1).float()

    a = torch.sum(input_flat * target_flat, 1)
    if naive_dice:
        b = torch.sum(input_flat, 1)
        c = torch.sum(target_flat, 1)
        dice = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input_flat * input_flat, 1) + eps
        c = torch.sum(target_flat * target_flat, 1) + eps
        dice = (2 * a) / (b + c)

    dice_loss = 1 - dice

    # Apply weights to Dice Loss
    if weight is not None:
        assert weight.ndim == dice_loss.ndim
        assert len(weight) == len(pred)
        dice_loss *= weight

    # Reduce Dice Loss
    if reduction == 'mean':
        dice_loss = dice_loss.mean()
    elif reduction == 'sum':
        dice_loss = dice_loss.sum()

    # Calculate Focal Loss
    focal_loss_value = focal_loss(pred, target, weight, gamma, alpha, reduction, avg_factor)

    # Combine the losses
    combined_loss = focal_weight * focal_loss_value + dice_weight * dice_loss

    # Apply average factor if provided
    if avg_factor is not None:
        combined_loss /= avg_factor

    return combined_loss



@MODELS.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-3,
                 use_mask=False):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        if self.use_mask:
            loss = self.loss_weight * combo_loss(
                pred,
                target,
                weight,
                eps=self.eps,
                reduction=reduction,
                naive_dice=self.naive_dice,
                avg_factor=avg_factor)
        else:
            loss = self.loss_weight * dice_loss(
                pred,
                target,
                weight,
                eps=self.eps,
                reduction=reduction,
                naive_dice=self.naive_dice,
                avg_factor=avg_factor)

        return loss

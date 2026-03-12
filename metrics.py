import torch
import torch.nn.functional as F


def multiclass_accuracy(y_pred, y_true):
    """
    y_pred: [B, C, H, W] logits
    y_true: [B, H, W] class indices (0..C-1)
    """

    # predicted class per pixel
    y_pred_labels = torch.argmax(y_pred, dim=1)   # [B,H,W]

    # compare prediction with ground truth
    acc = (y_pred_labels == y_true).float().mean(dim=(1,2))
    return acc

def multiclass_dice_score(y_pred, y_true, smooth=1e-6):
    """
    y_pred: [B, C, H, W] logits
    y_true: [B, H, W] class indices
    """
    # convert logits → predicted class
    y_pred_labels = torch.argmax(y_pred, dim=1)   # [B,H,W]
    num_classes = y_pred.shape[1]
    # convert to one-hot
    y_pred_onehot = F.one_hot(y_pred_labels, num_classes).permute(0,3,1,2).float()
    y_true_onehot = F.one_hot(y_true, num_classes).permute(0,3,1,2).float()

    intersection = (y_pred_onehot * y_true_onehot).sum(dim=(2,3))
    union = y_pred_onehot.sum(dim=(2,3)) + y_true_onehot.sum(dim=(2,3))

    dice = (2 * intersection + smooth) / (union + smooth)
    # average dice per image
    dice = dice.mean(dim=1)
    return dice

def multiclass_iou_score(y_pred, y_true, smooth=1e-6):
    """
    y_pred: [B, C, H, W] logits
    y_true: [B, H, W] class indices
    """

    # predicted class per pixel
    y_pred_labels = torch.argmax(y_pred, dim=1)   # [B,H,W]

    num_classes = y_pred.shape[1]

    # convert to one-hot
    y_pred_onehot = F.one_hot(y_pred_labels, num_classes).permute(0,3,1,2).float()
    y_true_onehot = F.one_hot(y_true, num_classes).permute(0,3,1,2).float()

    # intersection and union
    intersection = (y_pred_onehot * y_true_onehot).sum(dim=(2,3))
    union = y_pred_onehot.sum(dim=(2,3)) + y_true_onehot.sum(dim=(2,3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    # average IoU per image
    iou = iou.mean(dim=1)
    return iou


def multiclass_precision_recall(y_pred, y_true, num_classes):
    """
    y_pred: [B, C, H, W] logits
    y_true: [B, H, W] class indices
    """

    # predicted class
    y_pred_labels = torch.argmax(y_pred, dim=1)   # [B,H,W]

    # flatten
    y_pred_flat = y_pred_labels.view(-1)
    y_true_flat = y_true.view(-1)

    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)

    for cls in range(num_classes):
        TP[cls] = ((y_pred_flat == cls) & (y_true_flat == cls)).sum()
        FP[cls] = ((y_pred_flat == cls) & (y_true_flat != cls)).sum()
        FN[cls] = ((y_pred_flat != cls) & (y_true_flat == cls)).sum()

    return TP, FP, FN
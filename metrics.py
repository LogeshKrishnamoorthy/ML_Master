import torch
import torch.nn.functional as F

'''| Metric           | Type  
| ---------------- | ----- 
| Accuracy         | Micro  (its global level accuracy across batch, All pixels are equally important)
| Dice             | Macro  (Each class contributes equally)
| IoU              | Macro 
| Precision/Recall | Macro 
'''

def get_valid_preds_targets(y_pred, y_true, ignore_index):
    """
    Returns filtered predictions and targets (only valid pixels)

    y_pred: [B, C, H, W] logits
    y_true: [B, H, W] 
    """

    # squeeze if needed
    if y_true.dim() == 4:
        y_true = y_true.squeeze(1)

    # predicted labels
    y_pred_labels = torch.argmax(y_pred, dim=1)  # [B,H,W]

    # valid mask
    valid_mask = (y_true != ignore_index)

    # filter and flatten
    y_pred_valid = y_pred_labels[valid_mask]
    y_true_valid = y_true[valid_mask]

    return y_pred_valid, y_true_valid

def multiclass_accuracy(y_pred, y_true,ignore_index=None):
    # “All pixels across dataset vote equally”  --> Global accuracy across batch as like as micro  (total correct pixels / total pixels)  Dominated by background Misleading in imbalance
    # when you want per class level accuracy of each image use macro 
    ''''''
    # """
    # y_pred: [B, C, H, W] logits
    # y_true: [B, H, W] class indices (0..C-1)
    # """
    # predicted class per pixel
    if ignore_index is not None:
        y_pred, y_true = get_valid_preds_targets(y_pred, y_true, ignore_index)
    else: 
        y_pred = torch.argmax(y_pred, dim=1)  # [B,H,W]
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

    # compare prediction with ground truth
    correct = (y_pred == y_true).sum()
    total = y_true.numel()

    return correct, total


def multiclass_dice_score(y_pred, y_true,num_classes, ignore_index=None,smooth=1e-6):
    """
    y_pred: [B, C, H, W]
    y_true: [B, H, W]
    """
    # num_classes = y_pred.shape[1]

    # WITH ignore_index
    if ignore_index is not None:
        y_pred, y_true = get_valid_preds_targets(y_pred, y_true, ignore_index)
        # shapes: [N]

        dice_scores = torch.zeros(num_classes, device=y_pred.device)

        for cls in range(num_classes):
            pred_cls = (y_pred == cls)    # True where model predicted class `cls` # stick only valid classes in pred
            true_cls = (y_true == cls)

            intersection = (pred_cls & true_cls).sum().float()  # TRUE POSITIVES (TP) -> intersection
            
            total = pred_cls.sum() + true_cls.sum()
            '''pred_cls.sum() → predicted pixels of class cls (TP + FP)
            true_cls.sum() → actual pixels of class cls (TP + FN)'''

            if total > 0:   # formula 2TP/2TP+FP+FN 
                dice_scores[cls] = (2 * intersection + smooth) / (total + smooth)

        return dice_scores.mean()

    # WITHOUT ignore_index
    else:
        y_pred = torch.argmax(y_pred, dim=1)  # [B,H,W]

        # one-hot
        '''y_pred        : [B, H, W]
        one_hot       : [B, H, W, C]
        permute       : [B, C, H, W]'''
        y_pred_onehot = F.one_hot(y_pred, num_classes).permute(0,3,1,2).float()
        y_true_onehot = F.one_hot(y_true, num_classes).permute(0,3,1,2).float()

        intersection = (y_pred_onehot * y_true_onehot).sum(dim=(2,3)) # TP -> Element-wise multiply → keeps only overlapping pixels Sum over H, W → counts pixels
        union = y_pred_onehot.sum(dim=(2,3)) + y_true_onehot.sum(dim=(2,3))

        dice = (2 * intersection + smooth) / (union + smooth)

        return dice.mean()  

def multiclass_iou_score(y_pred, y_true,num_classes,  ignore_index=None, smooth=1e-6,):
    """
    y_pred: [B, C, H, W]
    y_true: [B, H, W]
    """
    # num_classes = y_pred.shape[1]

    # WITH ignore_index
    if ignore_index is not None:
        y_pred, y_true = get_valid_preds_targets(y_pred, y_true, ignore_index)
        # shapes: [N]

        iou_scores = torch.zeros(num_classes, device=y_pred.device)

        for cls in range(num_classes):
            pred_cls = (y_pred == cls)
            true_cls = (y_true == cls)

            intersection = (pred_cls & true_cls).sum().float()
            union = pred_cls.sum() + true_cls.sum() - intersection    # (TP+FP)+(TP+FN)−TP  =>  TP+FP+FN

            if union > 0:
                iou_scores[cls] = (intersection + smooth) / (union + smooth)

        return iou_scores.mean()

    # WITHOUT ignore_index
    else:
        y_pred_labels = torch.argmax(y_pred, dim=1)  # [B,H,W]

        # one-hot
        y_pred_onehot = F.one_hot(y_pred_labels, num_classes).permute(0,3,1,2).float()
        y_true_onehot = F.one_hot(y_true, num_classes).permute(0,3,1,2).float()

        intersection = (y_pred_onehot * y_true_onehot).sum(dim=(2,3))
        union = (
            y_pred_onehot.sum(dim=(2,3)) +
            y_true_onehot.sum(dim=(2,3)) -
            intersection
        )

        iou = (intersection + smooth) / (union + smooth)

        return iou.mean() 


def multiclass_precision_recall(y_pred, y_true, num_classes, ignore_index=None):
    y_pred = torch.argmax(y_pred, dim=1)

    if ignore_index is not None:
        mask = (y_true != ignore_index)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
    else:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)


    '''import torch
    x = torch.tensor([0, 1, 1, 2, 2, 2])
    counts = torch.bincount(x,min_length = 4)
    print(counts)   =>  tensor([1, 2, 3,0])'''

    cm = torch.bincount(
        num_classes * y_true + y_pred,    # num_classes * y_true + y_pred flattens (true, pred) pairs into unique indices,
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    TP = torch.diag(cm)
    FP = cm.sum(dim=0) - TP
    FN = cm.sum(dim=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    return TP, FP, FN
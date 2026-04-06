import os
import matplotlib.pyplot as plt
import torch

def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img * std + mean

def plot_batch_images_masks_preds(X, y_true, y_pred,n_classes,custom_cmap,check, save_path,max_samples=None,loss=None,iou=None,ignore_index=None):
    """
    Plots images, ground truth masks, and predicted masks for a batch
    and saves the figure as an image.

    Args:
        X: torch.Tensor [B,C,H,W] - images
        y_true: torch.Tensor [B,1,H,W] - ground truth masks
        y_pred: torch.Tensor [B,1,H,W] - predicted masks (already thresholded)
        max_samples: int, number of samples to plot from batch
        save_path: str, filename to save the figure
    """
    if X.dim() == 3:
        X = X.unsqueeze(0)
        y_true = y_true.unsqueeze(0)
        y_pred = y_pred.unsqueeze(0)

    batch_size = X.size(0)
    if max_samples is None:
        rows = batch_size
    else:
        rows = min(max_samples, batch_size)
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows), squeeze=False)
    fig.suptitle("Image | Ground Truth Mask | Predicted Mask", fontsize=14)

    for i in range(rows):

        # Do denormalization becuase it is normalized during transformation for similar scale across all channels so matplot expects 0-1 normalization but in that transformation we manually normalized with some values
        img = denormalize(X[i])      # after denormalize [-0.01, 1.02] becuase of interpolation so we use clamp
        img = img.clamp(0, 1)  # safety

        img_np = img.permute(1,2,0).cpu()
        mask_np = y_true[i].squeeze(0).cpu()   # imshow expects 2D for grayscale:
        pred_np = y_pred[i].squeeze(0).cpu()

        if ignore_index is not None:
            mask_np = mask_np.clone()
            pred_np = pred_np.clone()

            # Replace ignored pixels with background (0) for visualization
            mask_np[mask_np == ignore_index] = 0
            pred_np[pred_np == ignore_index] = 0

        # Image
        axes[i,0].imshow(img_np)
        axes[i,0].axis("off")
        axes[i,0].set_title("Image", fontsize=10)

        # Ground Truth Mask
        axes[i,1].imshow(mask_np, cmap=custom_cmap, vmin=0, vmax=n_classes)
        axes[i,1].axis("off")
        axes[i,1].set_title("Mask", fontsize=10)

        # Error mask
        # error_mask = (pred_np != mask_np)

        # Predicted Mask
        axes[i,2].imshow(pred_np, cmap=custom_cmap, vmin=0, vmax=n_classes)
        # axes[i,2].imshow(error_mask, cmap="Reds", alpha=0.5)
        axes[i,2].axis("off")
        axes[i,2].set_title("Predicted", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3)

    image_samples = "Train_Test_Samples"
    os.makedirs(image_samples, exist_ok=True)

    metrics_text = ""
    if loss is not None:
        metrics_text += f"{check} Loss: {loss:.4f}  "
    if iou is not None:
        metrics_text += f"IoU: {iou:.4f}  "
    if metrics_text:
        fig.text(
            0.50, 0.010,  # center bottom
            metrics_text,
            ha='center',
            fontsize=20
            # bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

    # Save figure to current working directory
    plt.savefig(os.path.join(os.getcwd(),image_samples, save_path), dpi=150) # dpi=50 → image looks pixelated, dpi=300 → publication-quality sharp image
    plt.close(fig)  # close figure to free memory

# Overlap in the same image
# import matplotlib.pyplot as plt
# import math
# def plot_overlay_samples(total_img, cols):
#     rows = math.ceil(total_img / cols)
#     plt.figure(figsize=(15, 5 * rows))
#     plt.suptitle("Image + Mask Overlay Samples", fontsize=18)

#     for i in range(total_img):
#         img, mask = train_dataset[i]

#         # Convert image CHW → HWC
#         img = img.permute(1, 2, 0).numpy()
#         mask = mask.squeeze(0).numpy()

#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(img)
#         plt.imshow(mask, cmap="jet", alpha=0.4)  # overlay
#         plt.title(train_dataset.images[i])
#         plt.axis("off")

#     plt.tight_layout()
#     plt.show()
# plot_overlay_samples(total_img=6, cols=3)


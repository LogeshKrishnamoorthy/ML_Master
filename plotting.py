import os
import matplotlib.pyplot as plt


def plot_batch_images_masks_preds(X, y_true, y_pred, save_path):
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
    batch_size = X.size(0)
    rows =  batch_size
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows), squeeze=False)
    fig.suptitle("Image | Ground Truth Mask | Predicted Mask", fontsize=14)

    for i in range(rows):
        img_np = X[i].permute(1,2,0).cpu()
        mask_np = y_true[i].squeeze(0).cpu()   # imshow expects 2D for grayscale:
        pred_np = y_pred[i].squeeze(0).cpu()

        # Image
        axes[i,0].imshow(img_np)
        axes[i,0].axis("off")
        axes[i,0].set_title("Image", fontsize=10)

        # Ground Truth Mask
        axes[i,1].imshow(mask_np, cmap="gray")
        axes[i,1].axis("off")
        axes[i,1].set_title("Mask", fontsize=10)

        # Predicted Mask
        axes[i,2].imshow(pred_np, cmap="gray")
        axes[i,2].axis("off")
        axes[i,2].set_title("Predicted", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3)

    image_samples = "Train_Test_Samples"
    os.makedirs(image_samples, exist_ok=True)

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


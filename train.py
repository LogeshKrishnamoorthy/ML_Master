import os
import datetime
from timeit import default_timer as timer  #  end - start
import numpy as np
from matplotlib.colors import ListedColormap
import torch
from torch import nn
from unet_multiclass import UNet_Multiclass
import torch.nn.functional as F
from tqdm import tqdm

from dataset import create_dataloaders
from dataset import get_train_transform    # data augumentation
from metrics import multiclass_accuracy,multiclass_dice_score,multiclass_iou_score,multiclass_precision_recall
from plotting import plot_batch_images_masks_preds
from checkpoint import save_checkpoint,load_checkpoint
from utilities import print_train_time



train_image_dir = r"C:\Users\logesh-INC-5718\Zoho\Tasks\Task5_Multi_Class_Segmentation\SegNet-Tutorial\CamVid\train"
train_mask_dir = r"C:\Users\logesh-INC-5718\Zoho\Tasks\Task5_Multi_Class_Segmentation\SegNet-Tutorial\CamVid\trainannot"
val_image_dir = r"C:\Users\logesh-INC-5718\Zoho\Tasks\Task5_Multi_Class_Segmentation\SegNet-Tutorial\CamVid\val"
val_mask_dir = r"C:\Users\logesh-INC-5718\Zoho\Tasks\Task5_Multi_Class_Segmentation\SegNet-Tutorial\CamVid\valannot"
test_image_dir = r"C:\Users\logesh-INC-5718\Zoho\Tasks\Task5_Multi_Class_Segmentation\SegNet-Tutorial\CamVid\test"
test_mask_dir = r"C:\Users\logesh-INC-5718\Zoho\Tasks\Task5_Multi_Class_Segmentation\SegNet-Tutorial\CamVid\testannot"


batch_size = 4
num_workers = 0
ignore_index = 255

train_loader = create_dataloaders(train_image_dir,train_mask_dir,batch_size,num_workers,transform=get_train_transform(),shuffle=True,aug_prob=0.50)
val_loader = create_dataloaders(val_image_dir,val_mask_dir,batch_size,num_workers,transform=None,shuffle=False)
test_loader = create_dataloaders(test_image_dir,test_mask_dir,batch_size,num_workers,transform=None,shuffle=False)

# if albumentation needed use this
# train_loader = create_dataloaders(train_image_dir,train_mask_dir,batch_size,num_workers,transform=get_train_transform())

# x_batch, y_batch = next(iter(train_loader))
# print("Image shape:", x_batch.shape)
# print("Mask shape:", y_batch.shape)

# For single batch 
# x_train, y_train,min, max,num_of_classes = next(iter(train_loader))
# print("Image shape:", x_train.shape)
# print("Mask shape:", y_train.shape)
# print("Minimum:",min)
# print("Maximum:",max)
# print("Total classes:",num_of_classes)


# ====================================
# Find number of classes in mask image
# ====================================
def compute_dataset_stats(loader,belongs):  # all over the batches 
    # iterate through all batches
    for _ in loader:
        pass
    dataset = loader.dataset
    print(f"Total Unique Classes in all {belongs} images: {dataset.classes}")
    # print("Global Minimum:", dataset.global_min)
    # print("Global Maximum:", dataset.global_max)
    # print("Total Classes:", len(dataset.classes))
    return dataset.global_max

n_classes_training = compute_dataset_stats(train_loader,belongs="training")+1  
n_classes_validation = compute_dataset_stats(val_loader,belongs="validation")+1
n_classes_testing = compute_dataset_stats(test_loader,belongs="testing")+1

n_classes = max(n_classes_training,n_classes_validation,n_classes_testing)
print("Maximum number of classes across Train, Validation, and Test datasets:", n_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
# It will only activate when CUDA is available.
if device == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN Optimization to accelerates convolution-heavy models like U-Net.



model_unet = UNet_Multiclass(in_channels=3, out_channels=n_classes).to(device=device)

log_file = os.path.join(os.getcwd(), "Metrics_Log_Train_Test.txt")



# Convert predictions of U-net into RGB format for class-to-color mapping
# Your class colors
cmap = np.array([
    [128,128,128],  # Sky(0) -> Medium Gray
    [128,0,0],      # Building(1) -> Maroon
    [192,192,128],  # Pole(2) -> yellowish gray
    [128,64,128],   # Road(3) ->  Deep Sky Blue
    [60,40,222],    # Pavement(4) -> Vivid Blue
    [128,128,0],    # Tree(5)   -> dark yellow color
    [192,128,128],  # SignSymbol(6) -> Faded Red.
    [64,64,128],    # Fence(7) -> Dark Moderate Blue    
    [64,0,128],     # Car(8) -> Deep Purple
    [64,64,0],      # Pedestrian(9) -> very dark, olive-green
    [0, 128, 192],    # ByCyclist(10) -> Magenta 
    [0, 0, 0]     # Others(11) -> black
], dtype=np.float32)
# Normalize [0,255] → [0,1]
cmap = cmap / 255.0
# Create matplotlib colormap
custom_cmap = ListedColormap(cmap)




# ==============================
# Loss function (Dice Loss)
# ==============================
# loss_fn = nn.CrossEntropyLoss()  

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        
        num_classes = logits.shape[1]
        # convert logits → probabilities
        probs = F.softmax(logits, dim=1)   # across classes [B, C, H, W]
        # remove channel dimension [B,1,H,W] -> grayscale 
        targets = targets.squeeze(1)          # [B,H,W]
        
        # ---------------------------
        # HANDLE IGNORE PIXELS
        # ---------------------------
        valid_mask = (targets != ignore_index)   # [B,H,W]  generate true for valid classes , false for 255 
        # replace ignore_index with 0 temporarily (safe for one_hot)
        targets_clone = targets.clone()
        targets_clone[~valid_mask] = 0  # apply false to  0 instead of 255 remaining true classes are same index 

        # convert target[B,H,W] → one-hot[B, H, W, C] 
        targets_one_hot = F.one_hot(targets_clone, num_classes)   # [B, H, W, C] 
        # targets =[[0,1],[2,1]]  ==> [[[1,0,0], [0,1,0]],[[0,0,1], [0,1,0]]]
        targets_one_hot = targets_one_hot.permute(0,3,1,2).float()  # [B, C, H, W]
        ''' targets:        [B,H,W]
            one-hot target: [B,C,H,W]
            predictions:    [B,C,H,W] '''

        # expand mask to match [B,C,H,W]
        valid_mask = valid_mask.unsqueeze(1)  # [B,1,H,W]

        # apply mask → remove ignored pixels
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # flatten --> [B,C,H,W] → [B,C,H*W] 
        # Dice loss can compute overlap across all pixels for each class efficiently.
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)

        # dice computation
        intersection = (probs * targets_one_hot).sum(dim=2)
        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()
    
# class_weights = torch.tensor([
#     0.5,  # Sky
#     1.0,  # Building
#     2.0,  # Pole (small)
#     1.0,  # Road
#     1.5,  # Pavement
#     1.0,  # Tree
#     2.0,  # SignSymbol (small)
#     1.5,  # Fence
#     1.5,  # Car
#     2.5,  # Pedestrian (small)
#     3.0,  # PersonWithCyclist (very small)
#     2.5   # Others
# ], dtype=torch.float32).to(device)

def compute_dynamic_class_weights(y, n_classes, eps=1e-6):
    # Batch-wise dynamic class weights
    y_flat = y.view(-1)
    y_flat = y_flat[y_flat != 255]  # remove ignore artifact pixels
    # class_counts = [1000, 100, 10]  # minlength = n_classes
    class_counts = torch.bincount(y_flat, minlength=n_classes).float()   # y_flat = [0,0,2,2,2]→ class_counts = [2,0,3]
    # '''class_counts = [1000, 0, 5] → weights = [0.001, HUGE, 0.2] ''' so we use clamp to stable training for all 1/0+0.001 = 1000 
    # Clamp to avoid extremely large weights (VERY IMPORTANT)
    class_counts = torch.clamp(class_counts, min=10)  # if 0 replace with 10
    # weights = [1/1000, 1/100, 1/10] = [0.001, 0.01, 0.1]
    class_weights = 1.0 / (class_counts + eps)
    # Normalize weights -> Keeps loss magnitude consistent ✅,without normalization -> Loss scale changes drastically per batch ❌Training becomes unstable ❌
    # [0.001, 0.01, 0.1] -> [0.027, 0.27, 2.7]  (sum ≈ n_classes)
    class_weights = class_weights / class_weights.sum() * n_classes 
    return class_weights.to(device)

# ce_loss = nn.CrossEntropyLoss(weight=class_weights)
dice_loss = DiceLoss()
def combined_loss(preds, targets,class_weights):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights,ignore_index=ignore_index) # ignore_index - Model won’t learn from artificial pixels include to 11th class # default mean for reduction field orelse change to sum
    ce = ce_loss(preds, targets)
    dice = dice_loss(preds, targets)
    return 0.5 * ce + 0.5 * dice # Sometimes Dice dominates training, so we balance it:
loss_fn = combined_loss


# ==============================
# Early Stopping --> stop training
# ==============================
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience  # Number of consecutive bad epochs allowed
        self.verbose = verbose  # Whether to print logs
        self.delta = delta    # Minimum improvement threshold
        self.counter = 0   # Counts how many epochs validation did NOT improve
        self.best_loss = float('inf')  # Initialize best loss as infinity
        self.early_stop = False # Flag to signal training should stop
 
    def __call__(self, val_loss, model, optimizer, epoch, checkpoint_path):
        if self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.counter = 0
            save_checkpoint(epoch, model, optimizer, val_loss, checkpoint_path)
 
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered")


early_stopper = EarlyStopping(patience=10, verbose=True)   # 10% of total epoch 



# ==============================
# Optimizer
# ==============================
optimizer = torch.optim.Adam(params=model_unet.parameters(), 
                            lr=1e-3,betas=(0.9,0.999),weight_decay=1e-4)
# slow down learning 👉 LR reduces when model stops improving
# Big steps(0.001) → fast learning,  Small steps(0.001*0.5) → slow learning, fine-tunes weights, fine-tunes weights
'''LR reduction is not to “learn more”,
it is to refine what is already learned, Yes, reducing LR to 0.0005 makes learning slower,
but it helps the model converge better and avoid overshooting.'''

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # because we monitor val_loss, min - lower value is better for val_loss , if val_iou use max
    factor=0.5,        # reduce LR by half
    patience=5        # wait 3 epochs before reducing, scheduler patience is less than early stopping patience here is 5 < 10 
)


# ==============================
# Training
# ==============================
def train_step(epoch,
               model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               multiclass_accuracy,
               plot_first_batch,  
               device: torch.device = device):
    model.train()
    train_loss, train_acc,train_dice,train_iou,total_samples = 0, 0,0,0,0
    total_correct, total_pixels = 0,0

    # For precision/recall
    TP_total = torch.zeros(n_classes, device=device)
    FP_total = torch.zeros(n_classes, device=device)
    FN_total = torch.zeros(n_classes, device=device)

    model.to(device)
    for  batch_idx, (X, y) in enumerate(tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}")):

        # 1. Send data to GPU
        X, y = X.to(device,non_blocking=True), y.to(device,non_blocking=True)

        batch_size = X.size(0)
        total_samples += batch_size

        # 2. Zero gradients
        optimizer.zero_grad()

        # 3. Forward pass (logits)
        y_pred = model(X)

        # resize prediction to match mask size
        y_pred = F.interpolate(
            y_pred,
            size=y.shape[-2:],   # (H,W) of mask
            mode="bilinear",
            align_corners=False
        )

        # Use When size mismatch for y, y_pred 
        # y = y.float()  # convert to float
        # y = F.interpolate(y, size=y_pred.shape[2:], mode='nearest')  # resize mask, Bilinear interpolation creates fractional values like: 0.34, 0.78, 0.12


        # 4. Compute loss (logits expected for LogLoss)
        class_weights = compute_dynamic_class_weights(y, n_classes)
        loss = loss_fn(y_pred, y,class_weights)    # Crossentropy is only apply for loss function not all becuase loss are LogLossMulticlass + Softmax

        # 5. Backward pass
        loss.backward()

        # 6. Optimizer step
        optimizer.step()

        
        # ---------------- Metrics ----------------
        # loss (per-batch average loss) → (scaled loss proportional to batch size)
        train_loss += loss.item() * batch_size   # averaging the loss over a batch reduces variance in gradients

        # accuracy  --> Accuracy always counts pixels with sum() for both binary and multiclass
        correct, total = multiclass_accuracy(y_pred, y,ignore_index)  # but here y_pred has not a sigmoid we add sigmoid manually
        total_correct += correct.item()
        total_pixels += total

        '''
        Binary segmentation:
            sum counts foreground pixels to compute intersection/union directly for Dice/IoU.
        Multiclass segmentation:
            mean averages the per-class or per-image metrics after computing them separately for each class/image.
        '''
        # Dice Score
        train_dice += multiclass_dice_score(y_pred,y,n_classes,ignore_index).item() * batch_size   # We want one Dice and IoU value for the batch, not one per class.

        # Iou score
        train_iou += multiclass_iou_score(y_pred,y,n_classes,ignore_index).item() * batch_size

        # Precision and recall
        TP, FP, FN = multiclass_precision_recall(y_pred, y,n_classes,ignore_index)
        # Accumulate TP, FP, FN for epoch-level precision/recall
        TP_total += TP
        FP_total += FP
        FN_total += FN

        y_pred_labels = torch.argmax(y_pred, dim=1) 

        # ---------------- Plot batches ----------------
        plot_batch_images_masks_preds(X[0:1], y[0:1], y_pred_labels[0:1],n_classes, custom_cmap,check = "Training", save_path=f"Current_Training_Batch_Samples.png",max_samples=1, loss=train_loss/total_samples,iou=train_iou/total_samples,ignore_index=ignore_index)

        plot_intial_final_during_training = ["Intial","Final"]
        for i in range(len(plot_intial_final_during_training)):
            if ((i==0 and batch_idx == 0) or (i==1 and batch_idx == int(len(data_loader))-2)) and plot_first_batch:
                plot_batch_images_masks_preds(X, y, y_pred_labels,n_classes, custom_cmap,check="Training", save_path=f"Epoch_{epoch}_Training_{plot_intial_final_during_training[i]}_Batch_Samples.png",max_samples=None, loss=train_loss/total_samples,iou=train_iou/total_samples,ignore_index=ignore_index)

        # ---------------- Logging ----------------
        if batch_idx % 25 == 0 or batch_idx == len(data_loader) - 1:
            # Running averages so far
            current_lr = optimizer.param_groups[0]['lr']
            current_loss = train_loss / total_samples
            current_acc = total_correct / total_pixels
            current_dice = train_dice / total_samples
            current_iou = train_iou / total_samples

            precision_per_class = TP_total / (TP_total + FP_total + 1e-6)
            recall_per_class = TP_total / (TP_total + FN_total + 1e-6)

            current_precision = precision_per_class.mean().item()
            current_recall = recall_per_class.mean().item()

            if batch_idx == 0:
                log_text = f"===== Epoch {epoch} =====\nTraining Log Metrics:\n"
            else:
                log_text = ""

            log_text  += (
                f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} | "
                f"Batch {batch_idx+1}/{len(data_loader)} | "
                f"Processed {total_samples}/{len(data_loader.dataset)} samples | "
                f"LR: {current_lr:.6f} | "
                f"Loss: {current_loss:.5f} | "
                f"Acc: {current_acc*100:.2f}% | "
                f"Dice: {current_dice:.4f} | "
                f"IoU: {current_iou:.4f} | "
                f"Prec: {current_precision:.4f} | "
                f"Rec: {current_recall:.4f}\n"
            )
            print(log_text)
            
            mode = "w" if (batch_idx == 0 and epoch == 1)  else "a"
            with open(log_file, mode) as f:
                f.write(log_text  + "\n")

    train_loss /= total_samples
    train_iou /= total_samples

    return train_loss, train_iou

# ==============================
# Validation and Testing
# ==============================
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              plot_first_batch,
              multiclass_accuracy,
              check,
              device: torch.device = device):
    test_loss, test_acc,test_iou,total_samples = 0, 0,0,0
    total_correct, total_pixels = 0,0
    
    # For precision/recall
    TP_total = torch.zeros(n_classes, device=device)
    FP_total = torch.zeros(n_classes, device=device)
    FN_total = torch.zeros(n_classes, device=device)   

    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for batch_idx,(X, y) in enumerate(tqdm(data_loader, total=len(data_loader), desc=check, leave=False)):
            # Send data to GPU
            X, y = X.to(device,non_blocking=True), y.to(device,non_blocking=True)
            
            # 1. Forward pass
            test_pred = model(X)

            test_pred = F.interpolate(
                test_pred,
                size=y.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            # y = y.float()
            # y = F.interpolate(y, size=test_pred.shape[2:], mode='nearest')

            '''If you resize mask using default interpolation (bilinear),   
            it will create float values (0.3, 0.7, etc.) and destroy class labels.'''
            
            # 2. Calculate loss and accuracy
            class_weights = compute_dynamic_class_weights(y, n_classes)
            loss = loss_fn(test_pred, y,class_weights)

            batch_size = X.size(0)
            total_samples += batch_size
            
            test_loss += loss.item() * batch_size
            correct, total = multiclass_accuracy(test_pred,y, ignore_index=None)
            total_correct += correct.item()
            total_pixels += total

            test_iou += multiclass_iou_score(test_pred,y,n_classes,ignore_index=None).mean().item() * batch_size  # returns per-class IoU, summing them may inflate the value.

            TP, FP, FN  = multiclass_precision_recall(test_pred, y, n_classes,ignore_index=None)
            # Accumulate TP, FP, FN for epoch-level precision/recall
            TP_total += TP
            FP_total += FP
            FN_total += FN
            precision_per_class = TP_total / (TP_total + FP_total + 1e-6)
            recall_per_class = TP_total / (TP_total + FN_total + 1e-6)

            current_precision = precision_per_class.mean().item()
            current_recall = recall_per_class.mean().item()

            y_pred_labels = torch.argmax(test_pred, dim=1) 

            plot_batch_images_masks_preds(X[0:1], y[0:1], y_pred_labels[0:1],n_classes, custom_cmap,check=f"{check}",save_path=f"Current_{check}_Batch_Samples.png",max_samples=1, loss=test_loss/total_samples,iou=test_iou/total_samples,ignore_index=None)
            if (batch_idx == 0) and plot_first_batch:
                plot_batch_images_masks_preds(X, y, y_pred_labels,n_classes, custom_cmap,check=f"{check}",save_path=f"{check}_Image_Mask_Pred_Samples.png",max_samples=None, loss=test_loss/total_samples,iou=test_iou/total_samples,ignore_index=None)
                

            if batch_idx % 10 == 0 or batch_idx == len(data_loader) - 1:
                if batch_idx == 0:
                    log_text = f"\n{check} Metrics:\n"
                else:
                    log_text = ""

                log_text += (
                    f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} | "
                    f"Batch {batch_idx+1}/{len(data_loader)} | "
                    f"Processed {int(total_samples)}/{len(data_loader.dataset)} samples | "
                    f"Loss: {(test_loss/total_samples):.5f} | "
                    f"Acc: {(total_correct/total_pixels)*100:.2f}% | "
                    f"IoU: {(test_iou/total_samples):.4f} | "
                    f"Prec: {current_precision:.4f} | "
                    f"Rec: {current_recall:.4f}\n"
                )

                print(log_text)

                with open(log_file, "a") as f:
                    f.write(log_text + "\n")

    test_loss /= total_samples
    test_iou /= total_samples

    return test_loss, test_iou



if __name__ == "__main__":
    torch.manual_seed(42)

    x_batch, y_batch = next(iter(train_loader))
    print("Image shape:", x_batch.shape)
    print("Mask shape:", y_batch.shape)

    # ==============================
    # Checkpoint Directory
    # ==============================

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")  # latest_checkpoint exists only as a string, not a real file.

    best_iou = 0
    start_epoch = 1

    # ==============================
    # Resume Training (if checkpoint exists)
    # ==============================

    if os.path.exists(latest_checkpoint):

        start_epoch, _ = load_checkpoint(   # optimizer and model weights are still restored from the checkpoint,so training will continue exactly as it left off.
            latest_checkpoint,
            model_unet,
            optimizer
        )

        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")

    # ==============================
    # Training Timer
    # ==============================

    train_time_start_on_gpu = timer()

    epochs = 100

    # ==============================
    # Training Loop
    # ==============================

    last_train_loss, last_train_iou = 0.0, 0.0
    last_val_loss, last_val_iou = 0.0, 0.0

    for epoch in tqdm(range(start_epoch, epochs+1)):

        print(f"Epoch: {epoch}\n---------")

        train_loss, train_iou = train_step(
            epoch=epoch,
            data_loader=train_loader,
            model=model_unet,
            loss_fn=loss_fn,
            optimizer=optimizer,
            plot_first_batch=True,
            multiclass_accuracy=multiclass_accuracy
        )

        # ==============================
        # Validation
        # ==============================
        val_loss,val_iou = test_step(
        data_loader=val_loader,
        model=model_unet,
        loss_fn=loss_fn,
        plot_first_batch=True,
        check="Validating",
        multiclass_accuracy=multiclass_accuracy
        )

        last_train_loss = train_loss
        last_train_iou = train_iou
        last_val_loss = val_loss
        last_val_iou = val_iou

        # ==============================        
        # SCHEDULER 
        # ==============================
        scheduler.step(val_loss)



        # ==============================
        # Save Latest Checkpoint (for resume)
        # ==============================

        save_checkpoint(
            epoch,
            model_unet,
            optimizer,
            train_loss,
            latest_checkpoint
        )

        # ==============================
        # Save Best Model
        # ==============================
        if val_iou > best_iou:
            best_iou = val_iou

            best_model_path = os.path.join(
                checkpoint_dir,
                "best_unet_model.pth"
            )

            torch.save(model_unet.state_dict(), best_model_path)

            log_text = f"Best Epoch is {epoch} and best model updated (IoU: {best_iou:.4f})"
            print(log_text)

            with open(log_file, "a") as f:
                f.write(log_text + "\n")

        # ==============================
        # Parent directory checkpoint
        # ==============================

        parent_dir = os.getcwd()

        # Remove old parent checkpoint
        for file in os.listdir(parent_dir):
            if file.startswith("latest_checkpoint_epoch_") and file.endswith(".pth"):
                os.remove(os.path.join(parent_dir, file))

        # Save new checkpoint with current epoch
        parent_checkpoint = os.path.join(
            parent_dir,
            f"latest_checkpoint_epoch_{epoch}.pth"
        )

        save_checkpoint(
            epoch,
            model_unet,
            optimizer,
            train_loss,
            parent_checkpoint
        )

        # ==============================
        # Save Epoch Checkpoint
        # ==============================

        epoch_checkpoint = os.path.join(
            checkpoint_dir,
            f"unet_epoch_{epoch}.pth"
        )

        save_checkpoint(
            epoch,
            model_unet,
            optimizer,
            train_loss,
            epoch_checkpoint
        )


        # ==================================
        # Early Stopping for continuous loss
        # ==================================
        early_stopper(
            val_loss,
            model_unet,
            optimizer,
            epoch,
            latest_checkpoint
        )
        if early_stopper.early_stop:
            print(f"Continous loss in validation data, so stopping early at epoch {epoch}")
            break

        # ==================================
        # If last epoch finished
        # ==================================
        if epoch == epochs:
            keep_file = "best_unet_model.pth"
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if item == keep_file:
                    continue   # keep best model
                if os.path.isfile(item_path):
                    os.remove(item_path)
            print("Training completed. All checkpoints removed except best model.")

    # ==============================
    # Training End Time
    # ==============================

    train_time_end_on_gpu = timer()


    # ==============================
    # Testing uses BEST model, not Last model
    # ==============================
    best_model_path = os.path.join(checkpoint_dir, "best_unet_model.pth")
    if os.path.exists(best_model_path):
        model_unet.load_state_dict(torch.load(best_model_path,map_location=device))
        print("Loaded best model for testing")
    else:
        print("Best model not found, using last model")

    model_unet.eval()
    last_test_loss, last_test_iou = test_step(
        data_loader=test_loader,
        model=model_unet,
        loss_fn=loss_fn,
        plot_first_batch=True,
        check="Testing",
        multiclass_accuracy=multiclass_accuracy
    )

    
    log = (
        f"\n\n\nOverall Train/Val/Test loss and Iou: \n"
        f"Train loss: {last_train_loss:.5f} | "
        f"Train IoU score: {last_train_iou:.4f} | "
        f"Validation loss: {last_val_loss:.5f} | "
        f"Validation IoU score: {last_val_iou:.4f} | "
        f"Test loss: {last_test_loss:.5f} | "
        f"Test IoU score: {last_test_iou:.4f} "
    )

    print(log)

    total_train_time_sec, total_train_time = print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)

    with open(log_file, "a") as f:
        f.write( log + "\n" + total_train_time  + "\n")


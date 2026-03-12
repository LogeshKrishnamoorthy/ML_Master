import os
import datetime
from timeit import default_timer as timer  #  end - start
import torch
from torch import nn
from unet_multiclass import UNet_Multiclass
import torch.nn.functional as F
from tqdm import tqdm

from dataset import create_dataloaders
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

train_loader = create_dataloaders(train_image_dir,train_mask_dir,batch_size,num_workers)
val_loader = create_dataloaders(val_image_dir,val_mask_dir,batch_size,num_workers)
test_loader = create_dataloaders(test_image_dir,test_mask_dir,batch_size,num_workers)

# For single batch 
# x_train, y_train,min, max,num_of_classes = next(iter(train_loader))
# print("Image shape:", x_train.shape)
# print("Mask shape:", y_train.shape)
# print("Minimum:",min)
# print("Maximum:",max)
# print("Total classes:",num_of_classes)


def compute_dataset_stats(loader):  # all over the batches 
    # iterate through all batches
    for _ in loader:
        pass
    dataset = loader.dataset
    print(dataset.classes)
    # print("Global Minimum:", dataset.global_min)
    # print("Global Maximum:", dataset.global_max)
    # print("Total Classes:", len(dataset.classes))
    return dataset.global_max

n_classes_training = compute_dataset_stats(train_loader)+1  

n_classes_validation = compute_dataset_stats(val_loader)+1

n_classes_testing = compute_dataset_stats(test_loader)+1

n_classes = max(n_classes_training,n_classes_validation,n_classes_testing)
print("Maximum number of classes across Train, Validation, and Test datasets:", n_classes)
device = "cuda" if torch.cuda.is_available() else "cpu"

# It will only activate when CUDA is available.
if device == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN Optimization to accelerates convolution-heavy models like U-Net.



model_unet = UNet_Multiclass(in_channels=3, out_channels=n_classes).to(device=device)

log_file = os.path.join(os.getcwd(), "Metrics_Log_Train_Test.txt")

loss_fn = nn.CrossEntropyLoss()  

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits)  # convert logits → probabilities
#         probs = probs.view(probs.size(0), -1)
#         targets = targets.view(targets.size(0), -1)
#         intersection = (probs * targets).sum(dim=1)
#         dice = (2. * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
#         return 1 - dice.mean()
    
# bce_loss = nn.CrossEntropyLoss()
# dice_loss = DiceLoss()
# def combined_loss(preds, targets):
#     bce = bce_loss(preds, targets)
#     dice = dice_loss(preds, targets)
#     return 0.5*bce + 0.5*dice  # Sometimes Dice dominates training, so we balance it:
# loss_fn = combined_loss

optimizer = torch.optim.Adam(params=model_unet.parameters(), 
                            lr=0.001)

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
            size=y.shape[1:],   # (H,W) of mask
            mode="bilinear",
            align_corners=False
        )

        # Use When size mismatch for y, y_pred 
        # y = y.float()  # convert to float
        # y = F.interpolate(y, size=y_pred.shape[2:], mode='nearest')  # resize mask, Bilinear interpolation creates fractional values like: 0.34, 0.78, 0.12


        # 4. Compute loss (logits expected for BCEWithLogits)
        loss = loss_fn(y_pred, y)    # sigmoid is only apply for loss function not all becuase loss are BCE + sigmoid

        # 5. Backward pass
        loss.backward()

        # 6. Optimizer step
        optimizer.step()

        
        # ---------------- Metrics ----------------
        # loss
        train_loss += loss.item() * batch_size

        # accuracy
        train_acc += multiclass_accuracy(y_pred, y).sum().item()  # but here y_pred has not a sigmoid we add sigmoid manually
        
        # Dice Score
        train_dice += multiclass_dice_score(y_pred,y).sum().item()

        # Iou score
        train_iou += multiclass_iou_score(y_pred,y).sum().item()

        # Precision and recall
        TP, FP, FN = multiclass_precision_recall(y_pred, y,n_classes)
        # Accumulate TP, FP, FN for epoch-level precision/recall
        TP_total += TP
        FP_total += FP
        FN_total += FN

        y_pred_labels = torch.argmax(y_pred, dim=1) 

        # ---------------- Plot batches ----------------
        plot_intial_final_during_training = ["Intial","Final"]
        for i in range(len(plot_intial_final_during_training)):
            if ((i==0 and batch_idx == 0) or (i==1 and batch_idx == int(len(data_loader))-2)) and plot_first_batch:
                plot_batch_images_masks_preds(X, y, y_pred_labels,save_path=f"Epoch_{epoch}_Training_{plot_intial_final_during_training[i]}_Batch_Samples.png")

        # ---------------- Logging ----------------
        if batch_idx % 25 == 0:
            # Running averages so far
            current_loss = train_loss / total_samples
            current_acc = train_acc / total_samples
            current_dice = train_dice / total_samples
            current_iou = train_iou / total_samples

            precision_per_class = TP_total / (TP_total + FP_total + 1e-6)
            recall_per_class = TP_total / (TP_total + FN_total + 1e-6)

            current_precision = precision_per_class.mean().item()
            current_recall = recall_per_class.mean().item()

            if batch_idx == 0:
                log_text = f"Training Log Metrics:\n===== Epoch {epoch} =====\n"
            else:
                log_text = ""

            log_text  += (
                f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} | "
                f"Batch {batch_idx+1}/{len(data_loader)} | "
                f"Processed {total_samples}/{len(data_loader.dataset)} samples | "
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

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= total_samples
    train_acc /= total_samples
    train_dice /= total_samples
    train_iou /= total_samples

    precision_per_class = TP_total / (TP_total + FP_total + 1e-6)
    recall_per_class = TP_total / (TP_total + FN_total + 1e-6)

    train_precision = precision_per_class.mean().item()
    train_recall = recall_per_class.mean().item()

    epoch_log = (f"\n\n\nOverall Training Metrics: \n" 
                 f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc*100:.4f}% | Train Dice Score: {train_dice:.4f} | Train IoU Score: {train_iou:.4f} | Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f}\n")
    print(epoch_log)

    with open(log_file, "a") as f:
        f.write(epoch_log + "\n")

    return train_loss, train_iou

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              plot_first_batch,
              multiclass_accuracy,
              device: torch.device = device):
    test_loss, test_acc,test_iou,total_samples = 0, 0,0,0
    # For precision/recall
    TP_total, FP_total, FN_total = 0, 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for batch_idx,(X, y) in enumerate(tqdm(data_loader, total=len(data_loader), desc="Testing", leave=False)):
            # Send data to GPU
            X, y = X.to(device,non_blocking=True), y.to(device,non_blocking=True)
            
            # 1. Forward pass
            test_pred = model(X)

            test_pred = F.interpolate(
                test_pred,
                size=y.shape[1:],
                mode="bilinear",
                align_corners=False
            )
            # y = y.float()
            # y = F.interpolate(y, size=test_pred.shape[2:], mode='nearest')

            '''If you resize mask using default interpolation (bilinear),   
            it will create float values (0.3, 0.7, etc.) and destroy class labels.'''
            
            # 2. Calculate loss and accuracy
            loss = loss_fn(test_pred, y)


            batch_size = X.size(0)
            total_samples += batch_size
            
            test_loss += loss.item() * batch_size
            test_acc += multiclass_accuracy(test_pred,y).sum().item()
            test_iou += multiclass_iou_score(test_pred,y).sum().item()

            TP, FP, FN  = multiclass_precision_recall(test_pred, y,n_classes)
            # Accumulate TP, FP, FN for epoch-level precision/recall
            TP_total += TP
            FP_total += FP
            FN_total += FN

            y_pred_labels = torch.argmax(test_pred, dim=1) 

            if (batch_idx == 0) and plot_first_batch:
                plot_batch_images_masks_preds(X, y, y_pred_labels,save_path=f"Testing_Image_Mask_Pred_Samples.png")
                

            if batch_idx % 25 == 0:
            
                log_text = (
                    f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} | "
                    f"Batch {batch_idx+1}/{len(data_loader)} | "
                    f"Processed {total_samples}/{len(data_loader.dataset)} samples | "
                    f"Loss: {(test_loss/total_samples):.5f} | "
                    f"Acc: {(test_acc/total_samples)*100:.2f}% | "
                    f"IoU: {(test_iou/total_samples):.4f} | "
                    f"Prec: {(TP_total/(TP_total+FP_total) if (TP_total+FP_total)!=0 else 0.0):.4f} | "
                    f"Rec: {(TP_total/(TP_total+FN_total) if (TP_total+FN_total)!=0 else 0.0):.4f}\n"
                )

                print(log_text)

                with open(log_file, "a") as f:
                    f.write(log_text + "\n")


    # After loop → overall test metrics
    test_loss /= total_samples
    test_acc /= total_samples
    test_iou /= total_samples
    test_precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) != 0 else 0.0
    test_recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) != 0 else 0.0

    test_epoch_log = (
        f"\n\n\nOverall Test Metrics: \n"
        f"Test loss: {test_loss:.5f} | "
        f"Test accuracy: {test_acc*100:.2f}% | "
        f"Test IoU score: {test_iou:.4f} | "
        f"Test Precision: {test_precision:.4f} | "
        f"Test Recall: {test_recall:.4f}\n"
    )

    print( test_epoch_log)

    with open(log_file, "a") as f:
        f.write(test_epoch_log + "\n")




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

    epochs = 30

    # ==============================
    # Training Loop
    # ==============================

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

        # ==============================
        # Save Best Model
        # ==============================

        if train_iou > best_iou:

            best_iou = train_iou

            best_model_path = os.path.join(
                checkpoint_dir,
                "best_unet_model.pth"
            )

            torch.save(model_unet.state_dict(), best_model_path)

            print(f"Best model updated (IoU: {best_iou:.4f})")

        # If last epoch finished
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
    # Testing
    # ==============================

    test_step(
        data_loader=test_loader,
        model=model_unet,
        loss_fn=loss_fn,
        plot_first_batch=True,
        multiclass_accuracy=multiclass_accuracy
    )


    total_train_time_sec, total_train_time = print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)

    with open(log_file, "a") as f:
        f.write(total_train_time  + "\n")


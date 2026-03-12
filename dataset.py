import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".png")]

        self.img_transform = transforms.Compose([
            transforms.Resize((360,480),
            interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((360,480),
            interpolation=InterpolationMode.NEAREST),
            transforms.PILToTensor()
        ])

         # statistics
        self.global_min = float("inf")
        self.global_max = float("-inf")
        self.classes = set()

    def __len__(self):
        return len(self.images)

    
    def __getitem__(self,index):

        img_name = self.images[index]
        img_path = os.path.join(self.image_dir,img_name)
        mask_path = os.path.join(self.mask_dir,img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = mask.squeeze(0).long()

        # update statistics
        m = mask.min().item()
        M = mask.max().item()

        self.global_min = min(self.global_min, m)
        self.global_max = max(self.global_max, M)

        self.classes.update(torch.unique(mask).tolist())


        return image, mask

def create_dataloaders(image_dir,mask_dir,batch_size,num_workers):
    
    dataset = SegmentationDataset(image_dir,mask_dir)

    data_loader = DataLoader(dataset,batch_size=batch_size,num_workers = num_workers,
                             shuffle=True,pin_memory=False,persistent_workers=False)

    return data_loader







# x_val, y_val = next(iter(val_loader))
# print("Image shape:", x_val.shape)
# print("Mask shape:", y_val.shape)

# # take first sample in batch
# img = x_val[0]   # [3, H, W]
# mask = y_val[0]  # [1, H, W]

# # convert to numpy and change format
# img = img.permute(1,2,0).numpy()   # [H,W,C]
# mask = mask.squeeze().numpy()      # [H,W]

# plt.figure(figsize=(20,10))

# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.title("Image")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(mask, cmap="gray")
# plt.title("Mask")
# plt.axis("off")

# plt.show()
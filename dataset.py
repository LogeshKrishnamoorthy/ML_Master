import os
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_base_transform():
    return A.Compose([
        A.Resize(360, 480,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),  # Normalize does NOT touch mask it normalize only in image;
        A.ToTensorV2(transpose_mask=False)    # image = [B,C,H,W], mask = [B,H,W] squeeze c in albumentation
        # If input is uint8 → yes, it scales to [0,1],  It always changes channel order
        # matplotlib draw only numpy array not tensor  use this line it through the error Invalid shape
    ])

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            # Less distortion → better learning
            scale=(0.95, 1.05),
            translate_percent=(0.05, 0.05), 
            rotate=(-10, 10),
            border_mode=0,
            fill=0,
            fill_mask=255, # Artifact pixels = 255 is the border
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3)  
    ])



class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir,aug_prob, 
                 base_transform,
                 aug_transform):
        
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        self.aug_prob = aug_prob

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

        # self.img_transform = transforms.Compose([
        #     transforms.Resize((360,480),
        #     interpolation=InterpolationMode.BILINEAR),
        #     transforms.ToTensor()
        # ])



        # self.mask_transform = transforms.Compose([
        #     transforms.Resize((360,480),
        #     interpolation=InterpolationMode.NEAREST),
        #     transforms.PILToTensor()
        # ])



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

        image = np.array(Image.open(img_path).convert("RGB"))  # [H,W,3]
        mask = np.array(Image.open(mask_path).convert("L"))     # [H,W]

        ## Albumentations internally use cv2 is NumPy arrays (cv2 format)
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(mask_path, 0)

        # if self.transform:
        #     augmented = self.transform(image=image, mask=mask)
        #     image = augmented["image"]
        #     mask = augmented["mask"]
        # else:
        #     image = self.img_transform(image)
        #     mask = self.mask_transform(mask)

         # Apply augmentation only sometimes
        if self.aug_transform and random.random() < self.aug_prob:
            augmented = self.aug_transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Always apply base transform after apply augmentation
        augmented = self.base_transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]   #image = [B,3,H,W], mask = [B,1,H,W]

        mask = mask.long()      # B,1,H,W
        # mask = mask.squeeze(0).long()   # remove channel 1 crossentropy expects that but it automatically handle albumentation totensorv2 => B,H,W    
        
        mask[mask > 11] = 255  # only 12 classes in this image more than this are considered as border artifact pixels as 255
        
        valid_mask = mask[mask != 255]     

        # update statistics
        m = valid_mask.min().item()
        M = valid_mask.max().item()

        self.global_min = min(self.global_min, m)
        self.global_max = max(self.global_max, M)

        self.classes.update(torch.unique(valid_mask).tolist())


        return image, mask

def create_dataloaders(image_dir,mask_dir,batch_size,num_workers,transform=None,shuffle=None,aug_prob=0.0):
    
    # dataset = SegmentationDataset(image_dir,mask_dir)
    dataset = SegmentationDataset(
        image_dir,
        mask_dir,
        aug_prob,   
        base_transform=get_base_transform(),
        aug_transform=transform,
    )


    data_loader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,
                             shuffle=shuffle,pin_memory=False,persistent_workers=False)

    return data_loader






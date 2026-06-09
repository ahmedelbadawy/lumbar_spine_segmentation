import cv2
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import Dataset

class LSS_Dataset(Dataset): # Custom PyTorch Dataset for loading lumbar ct image and corresponding masks

    def __init__(self, data_dir, images_files, masks_files, transform):
        self.data_dir = data_dir # Root directory where data is stored
        self.images_files = images_files # List of image filenames
        self.labels_files = masks_files # List of corresponding mask filenames
        self.transform = transform # Albumentations transform to apply on the image-mask pair

    def load_img(self, img_path):
        # Load a grayscale ct image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img
    
    def preprocess_mask(self, mask):
        # Map pixel values in the mask to class indices
        value_map = {0: 1, 100: 2, 200: 3, 250: 4, 255: 0}
        mask_out = np.zeros_like(mask, dtype=np.uint8)
        for pixel_val, class_idx in value_map.items():
            mask_out[mask == pixel_val] = class_idx
        return mask_out

    
    def load_mask(self, msk_path):
        # Load a mask and preprocess it to map pixel values to class labels
        msk = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        mask_out = self.preprocess_mask(msk)
        return mask_out

    def __getitem__(self, idx):
        # Sanity check: ensure that image and mask filenames match
        if self.images_files[idx][3:] != self.labels_files[idx][3:]:
            print(self.images_files[idx], self.labels_files[idx])
            raise AssertionError("Mask don't match image")

        img_path = os.path.join(self.data_dir,"T1_Output", self.images_files[idx])
        msk_path = os.path.join(self.data_dir,"Label_Images", self.labels_files[idx])

        # Load image and mask
        img = self.load_img(img_path)
        msk = self.load_mask(msk_path)

        # Apply augmentations and preprocessing
        augmented = self.transform(image = img, mask = msk)

        sample = {
            "image": augmented["image"].float(), # conver to float to match the model weights type
            "mask" : F.one_hot(augmented["mask"].long(), 5).permute(2, 0, 1).float()
            # One-hot encode the mask and reshape to (C, H, W) where c equal the number of classes
        }

        return sample
        
    def __len__(self):
 
        return len(self.images_files)

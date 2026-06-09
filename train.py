import wandb
from model import UNet
from datatset import LSS_Dataset
from engine import train_step, val_step
import torch
from tqdm import tqdm
import monai
import os 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc

def main():

    default_config = {
    "learning_rate": 3.18e-4,
    "optimizer": "AdamW",
    "scheduler_factor": 0.1066,
    "scheduler_patience": 8,

    "weighted_loss": False,
    "loss": "DiceLoss",

    "dropout": 0.3,
    "max_norm": 5,

    "se": 1,  # 0 = none, 1 = channel, 2 = channel + spatial

    "channels": [64, 128, 256, 512, 1024],
}
    
    wandb.init(
        project="#################",
        entity="##################",
        config=default_config
    )
    config = wandb.config
    # Set training EPOCHS
    NUM_EPOCHS = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    data_dir = "data"

    # reading the ct scans simages
    ct_images_files = sorted(os.listdir(os.path.join(data_dir,"T1_Output")))

    # reading the masks
    masks_files = sorted(os.listdir(os.path.join(data_dir,"Label_Images")))

    # determin the patient of each image, to split them based on patient
    patients = np.unique([file[3:7] for file in ct_images_files])

    # data split into 60% train, 20% val, and 20% test
    # Step 1: Split into train (60%) and temp (40%)
    train_patients, temp_patients = train_test_split(patients, test_size=0.4, random_state=42)
    # Step 2: Split temp into validation (20%) and test (20%)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.5, random_state=42)

    # function to filter files by patient group
    def filter_by_patients(file_list, patient_list):
        return [f for f in file_list if f[3:7] in patient_list]

    # split CT images and masks according to patient group
    train_images = filter_by_patients(ct_images_files, train_patients)
    val_images = filter_by_patients(ct_images_files, val_patients)
    test_images = filter_by_patients(ct_images_files, test_patients)

    train_masks = filter_by_patients(masks_files, train_patients)
    val_masks = filter_by_patients(masks_files, val_patients)
    test_masks = filter_by_patients(masks_files, test_patients)

    # Define training augmentations and normalization
    train_transform = A.Compose([
                                A.Resize(height=256, width=256, p=1, 
                                interpolation=cv2.INTER_LINEAR, 
                                mask_interpolation=cv2.INTER_NEAREST), # resize all images to 256 * 256
                                A.Affine( # Apply slight geometric transformations
                                rotate=(-10, 10),
                                scale=(0.95, 1.05),
                                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                                p=0.5 
                                ), 
                                A.Normalize(normalization = "min_max", p=1.0), # Normalize to [0, 1]
                                ToTensorV2() # Convert to PyTorch tensor
                                ])

    # Define validation transforms (no augmentation, just resize + normalize)
    val_transform = A.Compose([
                                A.Resize(height=256, width=256, p=1, 
                                interpolation=cv2.INTER_LINEAR, 
                                mask_interpolation=cv2.INTER_NEAREST),
                                A.Normalize(normalization = "min_max", p=1.0),
                                ToTensorV2()
                                ])

    # Create dataset instances with transforms
    train_dataset = LSS_Dataset(data_dir, train_images, train_masks, train_transform)
    val_dataset = LSS_Dataset(data_dir, val_images, val_masks, val_transform)
    test_dataset = LSS_Dataset(data_dir, test_images, test_masks, val_transform)

    # Initialize DataLoader for training, validation, and test sets

    train_loader = DataLoader(train_dataset, batch_size= 16)
    val_loader = DataLoader(val_dataset, batch_size= 16)
    test_loader = DataLoader(test_dataset, batch_size= 16)

    # Create model instance based on config
    model = UNet(1, 5, config.dropout,config.se, channels=config.channels)
    model = model.to(device)

    # Set up optimizer
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), config.learning_rate)
    
    # Enable mixed-precision training
    scaler = torch.GradScaler("cuda" ,enabled=True)
    
    # Learning rate scheduler (Reduce LR on Plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.scheduler_factor, patience=config.scheduler_patience, threshold=1e-4, min_lr = 1e-6, mode = "min", verbose = True)

    # Set class weights if using weighted loss, the weights were computed by taking the inverse of the each calss count and then normalize them 
    # to one without background
    weight = [0.0396, 0.0966, 0.5225, 0.3395] if config.weighted_loss else None 

    # Initialize loss function from MONAI
    loss_fn = getattr(monai.losses, config.loss)(weight=weight, include_background=  False)

    # start training loop
    for epoch in tqdm(range(NUM_EPOCHS)):
 
        ###########################Train##############################
        train_dice, avg_train_dice, train_loss = train_step(model=model,
                                                dataloader=train_loader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                scaler=scaler,
                                                max_norm=config.max_norm,
                                                device=device)
        ###########################val##############################
        val_dice, avg_val_dice, val_loss = val_step(model=model,
                                            dataloader=val_loader,
                                            loss_fn=loss_fn,
                                            device=device)
        # Update scheduler with validation loss
        scheduler.step(val_loss)
        
        # Log metrics to wandb
        fold_results = {
                "avg_train_dice": avg_train_dice,
                "avg_val_dice": avg_val_dice,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        wandb.log(fold_results)

        # Print metrics for current epoch
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
    
        print("\nTrain Dice Scores for each class:")
        for i, (class_name, score) in enumerate(train_dice.items()):
            print(f"Class {class_name} Dice Score: {score:.4f}")
        
        print(f"Average Train Dice Score: {avg_train_dice:.4f}")
    
        print("\nValidation Dice Scores for each class:")
        for i, (class_name, score) in enumerate(val_dice.items()):
            print(f"Class {class_name} Dice Score: {score:.4f}")
        
        print(f"Average Validation Dice Score: {avg_val_dice:.4f}")
        print("-" * 50)


    # Save trained model
    torch.save(model.state_dict(), "model.pth")
    # Log model as wandb artifact
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)
    
    ###############################################
    
    # Clean up at the end of training
    del model, optimizer, loss_fn, scaler
    gc.collect()
    torch.cuda.empty_cache()
    
    # Finish wandb run
    wandb.finish()
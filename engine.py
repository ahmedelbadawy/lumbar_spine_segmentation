import torch
from monai.metrics import DiceMetric
import torch.nn.functional as F
import gc

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               scaler: torch.amp.GradScaler,
               max_norm: int,
               device: torch.device
               ):
    """
    Trains a PyTorch model for one epoch.

    Args:
        model: The neural network model to train.
        dataloader: DataLoader providing training data.
        loss_fn: Loss function to compute the error.
        optimizer: Optimizer to update model weights.
        scaler: Gradient scaler for mixed precision training.
        max_norm: Max norm for gradient clipping.
        device: Device on which to perform computations.

    Returns:
        class_dice_scores: Dice scores for each class.
        class_dice_scores_mean: Mean Dice score.
        train_loss: Average loss for the epoch.
    """

    train_loss = 0
    # Dice metrics for per-class and average calculation
    train_dice_metric = DiceMetric(include_background=False, reduction  = "mean_batch", return_with_label=True)
    train_dice_metric_mean = DiceMetric(include_background=False, reduction  = "mean")
    
    model.train() # Set model to training mode
    for batch, sample in enumerate(dataloader):
        
        X  = sample["image"]
        y = sample["mask"]

        X, y = X.to(device), y.to(device) # move data to the same device of the model

        optimizer.zero_grad() # Reset gradients
        
        # Mixed precision forward pass for optimization
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            
            y_pred = F.softmax(model(X), dim=1) # Forward pass + softmax

            loss = loss_fn(y_pred, y) # Compute loss

        # accumulate loss across batches
        train_loss += loss.item() #

        # Loss backward
        scaler.scale(loss).backward()

        # Unscale gradients
        scaler.unscale_(optimizer)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Optimizer step
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
        
        # Convert predictions to one-hot encoding for Dice
        y_pred = y_pred.argmax(dim=1)
        y_pred = F.one_hot(y_pred, 5).permute(0, 3, 1, 2).float() 
        
        # Update Dice Metric
        train_dice_metric(y_pred, y)
        train_dice_metric_mean(y_pred, y)

    # Compute and log Dice score after each epoch for each class and mean
    class_dice_scores = train_dice_metric.aggregate()  # Get Dice score for each class
    train_dice_metric.reset()  # Reset the metric for the next epoch

    class_dice_scores_mean = train_dice_metric_mean.aggregate().item()  # Get Dice score for mean
    train_dice_metric_mean.reset()  # Reset the metric for the next epoch
    
    train_loss = train_loss/len(dataloader) # Average training loss

    # Clean up memory

    try:
        del X, y, y_pred, loss
    except:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return class_dice_scores, class_dice_scores_mean, train_loss
    

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> tuple[float, float]:
    """
    Validates a PyTorch model for one epoch.

    Args:
        model: The neural network model to validate.
        dataloader: DataLoader providing validation data.
        loss_fn: Loss function to compute the error.
        device: Device on which to perform computations.

    Returns:
        class_dice_scores: Dice scores for each class.
        class_dice_scores_mean: Mean Dice score.
        val_loss: Average validation loss for the epoch.
    """
    # Put model in eval mode
    model.eval() 

    # Setup val loss
    val_loss = 0

    # Dice metrics for validation
    val_dice_metric = DiceMetric(include_background=False, reduction  = "mean_batch", return_with_label=True)
    val_dice_metric_mean = DiceMetric(include_background=False, reduction  = "mean")

    # Disable gradient tracking
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, sample in enumerate(dataloader):

            X  = sample["image"]
            y = sample["mask"]

            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            y_pred = F.softmax(model(X), dim=1)

            # 2. Calculate and accumulate loss
            if loss_fn is not None:
                loss = loss_fn(y_pred, y)
        
                val_loss += loss.item()

            # Convert predictions to one-hot encoding for Dice
            y_pred = F.one_hot(y_pred.argmax(dim=1), 5).permute(0, 3, 1, 2).float() 

            # Update Dice metrics
            val_dice_metric(y_pred, y)
            val_dice_metric_mean(y_pred, y)

    # Aggregate Dice scores
    class_dice_scores = val_dice_metric.aggregate()  # Get Dice score for each class
    val_dice_metric.reset()  # Reset the metric for the next epoch

    class_dice_scores_mean = val_dice_metric_mean.aggregate().item()  # Get Dice score for mean
    val_dice_metric_mean.reset()  # Reset the metric for the next epoch
    
    val_loss = val_loss/len(dataloader) # Average validation loss

    # Clean up memory
    try:
        del X, y, y_pred, loss
    except:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return class_dice_scores, class_dice_scores_mean,val_loss
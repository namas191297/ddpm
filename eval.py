import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from model import DiffusionUNet
from ddpm_utils import get_train_test_dataloaders, convert_to_original, show_data_batch
from torch.utils.tensorboard import SummaryWriter
from config import training_config
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device):
    
    # Initialize test loss list
    test_losses = []

    # Use torch.no_grad to avoid updating gradients
    with torch.no_grad():

        for data in dataloader:

            # Get the batch
            _, x_noised, noise, t = data
            x_noised = x_noised.to(device)
            t = t.to(device)
            noise = noise.to(device)

            # Inference
            # Run it through the model to get the predicted noise
            predicted_noise = model(x_noised, t)

            # Calculate the loss between original noise and predicted noise
            predicted_noise = predicted_noise.view(predicted_noise.shape[0], -1)
            noise = noise.view(noise.shape[0], -1)
            loss = criterion(predicted_noise, noise)
            test_losses.append(loss.cpu().item())

        test_epoch_loss = np.mean(test_losses)
        return test_epoch_loss

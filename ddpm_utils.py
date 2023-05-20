import torch
import matplotlib.pyplot as plt
import numpy as np
from config import data_loader_config, dataset_config
from dataset import DiffusionDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from config import dataset_config, training_config

# For dataset containing images with 3 channels
reverse_transform = transforms.Compose([
    transforms.Lambda(lambda scaled_image: (scaled_image + 1) * 127.5),
    transforms.Lambda(lambda image: image.permute(1,2,0)),
    transforms.Lambda(lambda image: image.numpy().astype(np.uint8)),
    transforms.ToPILImage() # Make sure tensor shape is HxWxC
])

# Methods from the dataset.py class defined here for ease of access
def convert_to_original(img, channels=training_config['img_channels'], grid_save=False):
    # Apply transformation on image
    return reverse_transform(img)

def get_cumulative_alpha_prods(alphas):
    return torch.cumprod(alphas, dim=0)

def get_beta_scheduler(beta_scheduler_start, beta_scheduler_end, t):
    return torch.linspace(beta_scheduler_start, beta_scheduler_end, t)

# Initialize parameters based on the dataset
# Get the dataset parameters
T = dataset_config['t']
beta_scheduler_range = dataset_config['beta_scheduler_range']

# Parameter to be used for sampling
beta_scheduler_start, beta_scheduler_end = beta_scheduler_range
beta_scheduler = get_beta_scheduler(beta_scheduler_start, beta_scheduler_end, T)
alphas = 1. - beta_scheduler
alpha_prods = get_cumulative_alpha_prods(alphas)


# Utility methods
def show_data_batch(x, x_noised, t, show_images=3):
    rows = show_images
    cols = 2

    fig, ax = plt.subplots(rows,cols)
    for i in range(rows):
        for j in range(cols):
            if j == 0:
                ax[i,j].imshow(convert_to_original(x[i]), aspect='auto', interpolation='nearest')
            else:
                ax[i,j].imshow(convert_to_original(x_noised[i]), aspect='auto', interpolation='nearest')
                ax[i,j].set_title(f'Timestep:{t[i].item()}')
            ax[i,j].axis('off')
    fig.tight_layout()
    plt.show()

def get_train_test_dataloaders():
    
    train_ds = DiffusionDataset(images_dir = dataset_config['train_dir'],
                                image_size = dataset_config['image_size'],
                                t = dataset_config['t'],
                                beta_scheduler_range=dataset_config['beta_scheduler_range'])
    test_ds = DiffusionDataset(images_dir = dataset_config['test_dir'],
                                image_size = dataset_config['image_size'],
                                t = dataset_config['t'],
                                beta_scheduler_range=dataset_config['beta_scheduler_range'])
    
    # Create the dataloaders
    train_dl = DataLoader(train_ds, batch_size=data_loader_config['batch_size'], shuffle=data_loader_config['shuffle'])
    test_dl = DataLoader(test_ds, batch_size=data_loader_config['batch_size'], shuffle=data_loader_config['shuffle'])

    return train_dl, test_dl

@torch.no_grad()
def sample_datapoint(model, x, t, device='cuda'):

    # Initialize z
    if t > 0:
        z = torch.randn_like(x).to(device)
    else:
        z = 0

    # Apply inference to get the predicted noise
    epsilon_theta = model(x, t)

    # Get parameters at t'th index
    alphas_t = alphas[t].to(device)
    alpha_prods_t = alpha_prods[t].to(device)
    if t > 0:
        alphas_prods_t_prev = alpha_prods[t-1].to(device)
    else:
        alphas_prods_t_prev = 0
    betas_t = beta_scheduler[t].to(device)

    # Get intermediate result
    result = ((1/torch.sqrt(alphas_t)) * (x - (((1 - alphas_t)/(torch.sqrt(1 - alpha_prods_t))) * epsilon_theta)))
    if t > 0:
        posterior_variance = betas_t * ((1 - alphas_prods_t_prev)/(1 - alpha_prods_t))
    else:
        posterior_variance = 0

    # Get final resulting image
    result = result + posterior_variance * z
    return result
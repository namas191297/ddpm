import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from config import training_config
import ddpm_utils

class DiffusionDataset(Dataset):
    def __init__(self, images_dir, image_size=256, t=300, beta_scheduler_range=(1e-4, 0.02), channels=training_config['img_channels']):
        # Input Arguments:
        # image_dir -> Directory containing all the training images
        # image_size -> The size that the image needs to be resized to (int, eg. 512)
        # t -> timesteps (noise steps)
        # beta_scheduler_range -> tuple with (beta_start, beta_end)

        super().__init__()
        self.image_dir = images_dir
        self.images = os.listdir(images_dir)
        self.img_channels = channels
        self.image_size = image_size
        self.t = t
        self.beta_scheduler_start, self.beta_scheduler_end = beta_scheduler_range
        self.beta_scheduler = self.get_beta_scheduler()
        self.alphas = 1. - self.beta_scheduler
        self.alpha_prods = self.get_cumulative_alpha_prods()
        
        # Before applying diffusion
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            #transforms.RandomHorizontalFlip(), # ENABLE TO NON-MNIST DATA
            transforms.PILToTensor(),
            transforms.Lambda(lambda image: (image / 127.5) - 1)
        ])

    def get_cumulative_alpha_prods(self):
        return torch.cumprod(self.alphas, dim=0)

    def get_beta_scheduler(self):
        return torch.linspace(self.beta_scheduler_start, self.beta_scheduler_end, self.t)
    
    def apply_forward_diffusion(self, x_0, t):
        epsilon_noise = torch.randn_like(x_0)
        return (torch.sqrt(self.alpha_prods[t]) * x_0) + (torch.sqrt(1 - self.alpha_prods[t]) * epsilon_noise), epsilon_noise

    def __getitem__(self, index):
        
        # Read the image
        x = Image.open(os.path.join(self.image_dir, self.images[index]))

        # Transform the image (augmentation + scaling)
        if self.transform is not None:
            x = self.transform(x)

        # Get a random timestep
        t = torch.randint(0, self.t, (1,))
        
        # Apply forward diffusion to the image
        x_noised, noise = self.apply_forward_diffusion(x, t)
        
        # Return the original image, noised image, noise applied and timestep
        return x, x_noised, noise, t
    
    def __len__(self):
        return len(self.images)

if __name__ == '__main__':

    # Use this code to test
    ds = DiffusionDataset('data/sample_images')
    dl = DataLoader(ds, batch_size=10, shuffle=True)
    for data in dl:
        x, x_noised, noise, t = data
        ddpm_utils.show_data_batch(x, x_noised, t, show_images=3)
from model import DiffusionUNet
from ddpm_utils import convert_to_original, sample_datapoint
from config import dataset_config, training_config
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def generate_sample(model_ckpt, device='cuda'):

    # Create the model
    model = DiffusionUNet(img_channels=1, device=device).to(device) # Change image channels based on model, for MNIST ckpt it is 1
    model.load_state_dict(torch.load(model_ckpt))

    # Initialize variables fo sampling
    img = torch.randn((1, training_config['img_channels'], dataset_config['image_size'], dataset_config['image_size'])).to(device)
    T = dataset_config['t']

    for i in tqdm(range(0,T)[::-1]):
        t = torch.tensor([[i]]).to(device)
        img = sample_datapoint(model, img, t, device=device)
    
    # Save the image
    plt.imshow(convert_to_original(img.squeeze(0).detach().cpu()))
    plt.axis('off')
    plt.savefig('inference_output/output.png')
    plt.show()


if __name__ == '__main__':
    model_ckpt = 'training_checkpoints/test_model.ckpt'
    generate_sample(model_ckpt, 'cuda')

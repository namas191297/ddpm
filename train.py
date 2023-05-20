import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from model import DiffusionUNet
from ddpm_utils import get_train_test_dataloaders, convert_to_original, show_data_batch, sample_datapoint
from torch.utils.tensorboard import SummaryWriter
from config import training_config, dataset_config, data_loader_config
from eval import evaluate
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Initialize training variables
epochs = training_config['epochs']
lr = training_config['lr']
comment = f' batch_size = {data_loader_config["batch_size"]} lr = {training_config["lr"]} shuffle = {data_loader_config["shuffle"]}'
writer = SummaryWriter(comment=comment)
device = 'cuda' # or 'cpu'

# Initialize data loaders
train_dl, test_dl = get_train_test_dataloaders()

# Initialize model, optimizer and criterion
model = DiffusionUNet(device=device).to(device)
optimizer = Adam(params=model.parameters(),lr=lr)
criterion = nn.L1Loss()

# Training loop
lowest_loss = 1e13
for epoch in range(epochs):
    losses = []
    for data in tqdm(train_dl):

        _, x_noised, noise, t = data
        x_noised = x_noised.to(device)
        t = t.to(device)
        noise = noise.to(device)

        # Run it through the model to get the predicted noise
        predicted_noise = model(x_noised, t)

        # Calculate the loss between original noise and predicted noise
        predicted_noise = predicted_noise.view(predicted_noise.shape[0], -1)
        noise = noise.view(noise.shape[0], -1)
        loss = criterion(predicted_noise, noise)
    
        losses.append(loss.cpu().item())

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Log the training loss
    epoch_loss = np.mean(losses)
    # Append loss to calculate epoch loss
    if epoch_loss < lowest_loss:
        torch.save(model.state_dict(), f'training_checkpoints/model_{epoch_loss}_{epoch}_checkpoint.ckpt')
        lowest_loss = epoch_loss

    print(f'Epoch:{epoch+1}, Train Loss:{epoch_loss}')
    writer.add_scalar('Loss/train', epoch_loss, epoch)

    # Evaluate model
    test_epoch_loss = evaluate(model, test_dl, criterion, device)
    print(f'Epoch:{epoch+1}, Test Loss:{test_epoch_loss}')
    writer.add_scalar('Loss/test', test_epoch_loss, epoch)
    
    # Sample datapoints
    img = torch.randn((1, training_config['img_channels'], dataset_config['image_size'], dataset_config['image_size'])).to(device)
    T = dataset_config['t']
    images_to_plot = 5
    interval = T//images_to_plot
    plt.figure(figsize=(15,15))
    plt.axis('off')
    #outputs = []

    print('Plotting samples..')
    for i in tqdm(range(0,T)[::-1]):
        t = torch.tensor([[i]]).to(device)
        img = sample_datapoint(model, img, t, device=device)
        if i % interval == 0:
            #outputs.append(img.squeeze(0))
            plt.subplot(1, images_to_plot, i//interval+1)
            plt.title(f'Step:{i+1}')
            plt.axis('off')
            plt.imshow(convert_to_original(img.squeeze(0).detach().cpu()))
    
    #grid = torchvision.utils.make_grid(outputs[::-1])
    #torchvision.utils.save_image(grid, f'training_outputs/sample_{epoch+1}.png')
    plt.savefig(f'training_outputs/sample_{epoch+1}.png')
    plt.close()

writer.close()   
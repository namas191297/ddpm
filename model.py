import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import math
from torch.nn.functional import relu
from ddpm_utils import get_train_test_dataloaders
from einops import repeat
from torchvision.transforms.functional import resize
from config import training_config

# This is the diffusion model, which takes in a batch of images (with noise) and time step t,
# and returns the predicted noise. It is a UNet
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, up=False, device='cuda'):
        super().__init__()

        self.device = device
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.up = up
        self.time_linear = None

        if time_dim is not None:
            self.time_linear = nn.Linear(time_dim, out_channels)
        
        if up:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, t):
        
        # Project the input time embeddings into the output channel so that they can be added
        if self.time_linear is not None and t is not None:
            time_embeds = self.time_linear(t)
            time_embeds.squeeze(1)
            time_embeds = time_embeds.to(self.device) 

        # First convolution
        if self.up:
            out_1 = relu(self.bn(self.conv(x)))
        else:
            out_1 = relu(self.bn1(self.conv1(x)))
            

        # Add the time embedding to this output
        if self.time_linear is not None:
            repeated_time_embeds = torch.zeros((time_embeds.shape[0], self.out_channels, out_1.shape[2], out_1.shape[3])).to(self.device)
            for i in range(time_embeds.shape[0]):
                repeated_time_embeds[i] = repeat(time_embeds[i], '1 f -> f h w', h=out_1.shape[2], w=out_1.shape[3])
            out_1 = out_1 + repeated_time_embeds
        
        # Second convolution
        if self.up:
            out_2 = out_1
        else:
            out_2 = relu(self.bn2(self.conv2(out_1)))

        return out_2

class DiffusionUNet(nn.Module):
    def __init__(self, img_channels=training_config['img_channels'], out_channels=training_config['img_channels'], time_embeding_dim=32, device='cuda'):
        super().__init__()

        # Initialize parameters
        self.time_embedding_dim = time_embeding_dim
        self.downsample_channels = [64, 128, 256, 512, 1024]
        self.upsample_channels = [1024, 512, 256, 128, 64]

        # Sinusodial Positional Embedding Layer
        self.time_mlp = SinusoidalPosEmb(time_embeding_dim)

        # MaxPool Layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Downstream Layers
        self.resblockdown1 = ResBlock(img_channels, self.downsample_channels[0], time_dim=time_embeding_dim)
        self.resblockdown2 = ResBlock(self.downsample_channels[0], self.downsample_channels[1], time_dim=time_embeding_dim)
        self.resblockdown3 = ResBlock(self.downsample_channels[1], self.downsample_channels[2], time_dim=time_embeding_dim)
        self.resblockdown4 = ResBlock(self.downsample_channels[2], self.downsample_channels[3], time_dim=time_embeding_dim)
        self.bottleneck = ResBlock(self.downsample_channels[3], self.downsample_channels[4], time_dim=None)

        # Upstream layers
        self.resblockup1 = ResBlock(self.upsample_channels[0], self.upsample_channels[1], time_dim=None, up=True)
        self.updoubleconv1 = ResBlock(self.upsample_channels[0], self.upsample_channels[1], time_dim=time_embeding_dim)

        self.resblockup2 = ResBlock(self.upsample_channels[1], self.upsample_channels[2], time_dim=None, up=True)
        self.updoubleconv2 = ResBlock(self.upsample_channels[1], self.upsample_channels[2], time_dim=time_embeding_dim)

        self.resblockup3 = ResBlock(self.upsample_channels[2], self.upsample_channels[3], time_dim=None, up=True)
        self.updoubleconv3 = ResBlock(self.upsample_channels[2], self.upsample_channels[3], time_dim=time_embeding_dim)

        self.resblockup4 = ResBlock(self.upsample_channels[3], self.upsample_channels[4], time_dim=None, up=True)
        self.updoubleconv4 = ResBlock(self.upsample_channels[3], self.upsample_channels[4], time_dim=time_embeding_dim)

        # Output 1x1
        self.output1x1 = nn.Conv2d(self.upsample_channels[4], out_channels, kernel_size=1)

    def forward(self, x, t):
        
        # Get the time encodings
        time_encodings = self.time_mlp(t)

        # Forward - Downstream
        resop1 = self.resblockdown1(x, time_encodings)
        x = self.maxpool(resop1)
        resop2 = self.resblockdown2(x, time_encodings)
        x = self.maxpool(resop2)
        resop3 = self.resblockdown3(x, time_encodings)
        x = self.maxpool(resop3)
        resop4 = self.resblockdown4(x, time_encodings)
        x = self.maxpool(resop4)
        x = self.bottleneck(x, None)

        # Forward - Upstream
        x = self.resblockup1(x, None)
        if x.shape != resop4.shape:
            resop4 = resize(resop4, size=x.shape[2:])
        x = torch.cat([resop4, x], dim=1)
        x = self.updoubleconv1(x, time_encodings)

        x = self.resblockup2(x, None)
        if x.shape != resop3.shape:
            resop3 = resize(resop3, size=x.shape[2:])
        x = torch.cat([resop3, x], dim=1)
        x = self.updoubleconv2(x, time_encodings)

        x = self.resblockup3(x, None)
        if x.shape != resop2.shape:
            resop2 = resize(resop2, size=x.shape[2:])
        x = torch.cat([resop2, x], dim=1)
        x = self.updoubleconv3(x, time_encodings)

        x = self.resblockup4(x, None)
        if x.shape != resop1.shape:
            resop1 = resize(resop1, size=x.shape[2:])
        x = torch.cat([resop1, x], dim=1)
        x = self.updoubleconv4(x, time_encodings)

        # 1x1
        x = self.output1x1(x)
        return x
    
    def sample_data(self):
        pass
        
if __name__ == '__main__':
    train_dl, test_dl = get_train_test_dataloaders()
    device = 'cuda'
    model = DiffusionUNet(device=device).to(device)
    for data in train_dl:
        x, x_noised, noise, t = data
        x_noised = x_noised.to(device)
        t = t.to(device)
        op = model(x_noised, t)
        print(f'Input Noise:{noise.shape}, Predicted Noise:{op.shape}')
        break
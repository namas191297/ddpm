
# Denoising Diffusion Probabilistic Models

This repository is my implementation of the unconditional DDPM model presented in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). It contains a very simple implementation of the unconditional U-Net used to generate the noise residual without most of the modifications presented in the paper. The noise scheduler is built inside the dataset itself.The U-Net in this repo includes:

- Convolutional Residual Blocks in the U-Net.
- Fusing time-step sinusoidal position embedding into each residual block.

It does NOT include the following modifications(given in the paper):

- The U-Net is not based on Wide ResNet.
- No weight initialization.
- No dropout.

You can find most of the other parameters for the model in the config.py file. 




## Installation

1.Clone this repository.

2.Install the following dependencies:

```console
torch 
torchvision 
tensorboard 
tqdm 
numpy
matplotlib 
eionops 
pillow
```

3.Create the following empty folders:
1. data
2. data/train/
3. data/test/
4. inference_output
5. training_checkpoints
6. training_outputs

## Training

### Data

To setup your own training data:

- Place all the images you want to train on under data/train/
- Place all the images to be used for validation under data/test/
- If the images are colored (RGB), edit the config.py file and change the following:

```python
dataset_config = {
    ...
    ...
    'image_size':256, # or 128/512 based on your GPU.
    ...
    ...
}

training_config = {
    ...
    ...
    'img_channels':3 
}
```

Feel free to play around with other parameters in the config.py file, such as the learning rate, batch-size and t.

After making the changes, to train:
```bash
python train.py
```

After every epoch, the model will generate a sample containing 5 images containing the denoised images saved in training_outputs/ (denoised from right to left):

![git_images/example.png](Example of model trained on MNIST)



## Inference

Models get saved at every epoch under the trained_checkpoints folder. You can find the best model by looking at the smallest number after 'model_' in the saved ckpts.
For inference, open the inference.py file and make the following changes at the bottom of the file:

```python3
if __name__ == '__main__':
    model_ckpt = 'path_to_checkpoint' # Change this to where your .ckpt file is located.
    generate_sample(model_ckpt, 'cuda') # Change 'cuda' to 'cpu' if you don't have a GPU.
```

Make sure that config.py for the checkpoint is the same for inference as that when it was trained. 

The output will be saved in inference_output/ directory.


## Results

![git_images/result.jpg](Good quality results from the model trained on MNIST)
## Acknowledgements

- Code is heavily inspired by [Diffusion Models | PyTorch Implementation](https://www.youtube.com/watch?v=TBCRlnwJtZU)

- Sinusodial Embeddings Implementation: [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger)

- Thanks to the original paper!



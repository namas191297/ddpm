dataset_config = {
    'train_dir':'data/MNIST/train',
    'test_dir':'data/MNIST/test',
    'image_size':32, # 256 for RGB images
    't':300,
    'beta_scheduler_range':(1e-4, 0.02)
}

data_loader_config = {
    'batch_size':5, # Depending on your GPU 
    'shuffle':True,
}

training_config = {
    'epochs':100,
    'lr':2e-4,
    'img_channels':1 # 1 for grayscale images, 3 for RGB
}
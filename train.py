import os
os.environ["CUDA_VISIBLE_DEVICES"]="7,1,2,3,4,5,0,6"
import argparse
import multiprocessing
from PIL import Image
from pathlib import Path

import torch
import pytorch_lightning as pl
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from byol_pytorch import BYOL

# ssl model, a resnet 18

resnet = models.resnet18(pretrained=True) # True can get better performance.

# arguments

parser = argparse.ArgumentParser(description = 'byol-pytorch-lightning')

parser.add_argument('--image_folder', type = str, required = True, 
	help = 'path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants

BATCH_SIZE = 64
EPOCHS     = 300
LR         = 3e-4
NUM_GPUS   = 8
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = NUM_GPUS # or multiprocessing.cpu_count() 

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

# main

if __name__ == '__main__':
    ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(
    	ds, 
    	batch_size = BATCH_SIZE, 
    	num_workers = NUM_WORKERS, 
    	drop_last = True,
    	shuffle = True
    )

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 2048,
        moving_average_decay = 0.99
    )
    
    logger = TensorBoardLogger(
        save_dir = 'lightning_logs',
        name = 'logs'
    )

    checkpoint_callback = ModelCheckpoint(
        period = 10,
        save_top_k = -1
    )

    trainer = pl.Trainer(
       	gpus = NUM_GPUS,
        distributed_backend = 'ddp',
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        callbacks = [checkpoint_callback],
        logger = logger,
        # resume_from_checkpoint = '*.ckpt'
    )

    trainer.fit(model, train_loader)

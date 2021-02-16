import torch
import argparse
import pytorch_lightning as pl

from byol_pytorch import BYOL
from torchvision import models

parser = argparse.ArgumentParser(description = 'byol-lightning-test')

parser.add_argument('--ckpt_path', type = str, required = True,
    help = 'pytorch lightning checkpoint path')

parser.add_argument('--save_path', type = str, required = True,
    help = 'path to save pytorch checkpoint')

parser.add_argument('--arch', type = str, required = True,
    help = 'model arch')

args = parser.parse_args()

arch_dict = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101
}

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log('loss', loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


def convert_model(ckpt_path, save_path, arch):
    net = arch(pretrained=False)
    
    model = SelfSupervisedLearner(
        net,
        image_size = 256,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 2048,
        moving_average_decay = 0.99
    )
    
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    
    torch.save(net.state_dict(), save_path)

convert_model(args.ckpt_path, args.save_path, arch_dict[args.arch])

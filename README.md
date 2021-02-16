<img src="./BYOL.png" width="700px"></img>

## Bootstrap Your Own Latent (BYOL), in Pytorch

https://github.com/lucidrains/byol-pytorch 

This code repo is modified based on byol-pytorch 0.5.2.

The main changes include:

1. Added Checkpoints module.
2. Added Checkpoints conversion module (the saved Pytorch Lightning checkpoint can be convert as Pytorch checkpoint mode.)

## Install

```bash
$ pip install pytorch-lightning
$ pip install pillow
```

## Usage

Use Single or Multi-GPU with Pytorch-lightning: 

```python
# Training
$ python train.py --image_folder /path/to/your/images

# Ckpt convert
$ python ckpt_convert.py --ckpt_path *.ckpt --save_path *.pth --arch resnet*
```

## BYOL → SimSiam

You only need turn off turn off momentum in `train.py`.

```python
learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool',
    use_momentum = False       # turn off momentum in the target encoder
)
```

## Citation

```bibtex
@misc{grill2020bootstrap,
    title = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
    author = {Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
    year = {2020},
    eprint = {2006.07733},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@misc{chen2020exploring,
    title={Exploring Simple Siamese Representation Learning}, 
    author={Xinlei Chen and Kaiming He},
    year={2020},
    eprint={2011.10566},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

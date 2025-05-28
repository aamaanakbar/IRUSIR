# Investigating Robustess of  Unsupervised StyleGAN Image Restoration
### [[Arxiv]]() [[Website]](https://aamaanakbar.github.io/investigating_rusir/)

This paper `Investigating Robustess of  Unsupervised StyleGAN Image Restoration` will be presented at ICIP 2025. 

## Installation

1) First install the same environment as https://github.com/NVlabs/stylegan2-ada-pytorch.git. It is not essential for the custom cuda kernels to compile correctly, they just make things run ~30% faster.

2) Run `pip install tyro`. For running the evaluation you will also need to `pip install torchmetrics git+https://github.com/jwblangley/pytorch-fid.git`.

2) Download the pretrained StyleGAN model:
```bash
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -O pretrained/ffhq.pkl
```
## Restoring images

To run the tasks presented in the paper, use:

```bash 
python run.py --dataset_path datasets/samples
```

Some sample images have already been provided in `datasets/samples`.



















This code is borrowed from [robust unsupervised stylegan image restoration](https://github.com/yohan-pg/robust-unsupervised)
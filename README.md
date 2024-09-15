# Robust Unsupervised StyleGAN Image Restoration
### [[Arxiv]](https://arxiv.org/abs/2302.06733) [[Website]](https://lvsn.github.io/RobustUnsupervised/)

Code for the paper `Robust Unsupervised StyleGAN Image Restoration` presented at CVPR 2023. 

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


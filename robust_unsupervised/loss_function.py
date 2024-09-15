
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from lpips import LPIPS
from pytorch_msssim import ssim
from torchvision import models

# Pre-trained VGG model for feature extraction
class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, layers):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.model = torch.nn.Sequential(*[vgg[i] for i in layers]).eval().cuda()
    
    def forward(self, x):
        return self.model(x)

class MultiscaleLPIPS:
    def __init__(
        self,
        min_loss_res: int = 16,
        level_weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ssim_weight: float = 0.2,
        l1_weight: float = 0.1,
        consistency_weight: float = 0.3,
        feature_weight: float = 0.2,
        gradient_weight: float = 0.1
    ):
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.consistency_weight = consistency_weight
        self.feature_weight = feature_weight
        self.gradient_weight = gradient_weight
        
        self.lpips_network = LPIPS(net="vgg", verbose=False).cuda()
        self.feature_extractor = VGGFeatureExtractor(layers=[0, 5, 10, 19, 28])
        
    def measure_lpips(self, x, y, mask):
        if mask is not None:
            noise = (torch.randn_like(x) + 0.5) / 2.0
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)
        return self.lpips_network(x, y, normalize=True).mean() 

    def ssim_loss(self, pred, target):
        min_size = min(pred.size(-2), pred.size(-1))
        win_size = min(11, min_size)
        win_size = win_size - 1 if win_size % 2 == 0 else win_size
        return 1 - ssim(pred, target, data_range=1.0, size_average=True, win_size=win_size)

    def gradient_loss(self, pred, target):
        pred_grad = torch.abs(F.l1_loss(pred[:, :, 1:, :], pred[:, :, :-1, :], reduction='mean') +
                             F.l1_loss(pred[:, :, :, 1:], pred[:, :, :, :-1], reduction='mean'))
        target_grad = torch.abs(F.l1_loss(target[:, :, 1:, :], target[:, :, :-1, :], reduction='mean') +
                                F.l1_loss(target[:, :, :, 1:], target[:, :, :, :-1], reduction='mean'))
        return F.l1_loss(pred_grad, target_grad)

    def feature_loss(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return F.l1_loss(pred_features, target_features)

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask: Optional[Tensor] = None):
        x = f_hat(x_clean)
        losses = []

        if mask is not None:
            mask = F.interpolate(mask, y.shape[-1], mode="area")

        x_perturbed = x_clean + torch.randn_like(x_clean) * 0.01
        x_perturbed = f_hat(x_perturbed)
        consistency_loss = F.l1_loss(x, x_perturbed)
        x_perturbed = F.interpolate(x_perturbed, size=y.shape[-2:], mode='bilinear', align_corners=False)

        for weight in self.weights:
            if y.shape[-1] <= self.min_loss_res:
                break

            if weight > 0:
                loss_x = self.measure_lpips(x, y, mask)
                loss_x_perturbed = self.measure_lpips(x_perturbed, y, mask)
                symmetric_loss = (loss_x + loss_x_perturbed) / 2.0
                losses.append(weight * symmetric_loss)

            if mask is not None:
                mask = F.avg_pool2d(mask, 2)

            x = F.avg_pool2d(x, 2)
            x_clean = F.avg_pool2d(x_clean, 2)
            y = F.avg_pool2d(y, 2)
            x_perturbed = F.avg_pool2d(x_perturbed, 2)

        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else 0.0
        l1 = self.l1_weight * F.l1_loss(x, y)
        ssim = self.ssim_weight * self.ssim_loss(x, y)
        feature = self.feature_weight * self.feature_loss(x, y)
        gradient = self.gradient_weight * self.gradient_loss(x, y)
        total_loss = total + l1 + ssim + consistency_loss * self.consistency_weight + feature + gradient

        return total_loss
import torch
import torch.nn.functional as F
from math import exp
import cv2 
from ignite.metrics import PSNR, SSIM

def denormalize(z, mu, sigma):
    z = z*sigma+mu
    return z

def heatmap(F_frame, target_frame):
    diff_map = torch.abs(F_frame - target_frame).squeeze()
    # Normalize to 0 ~ 255.
    diff_map -= diff_map.min()
    diff_map /= diff_map.max()
    diff_map *= 255

    diff_map = diff_map.cpu().detach().numpy().astype('uint8')
    heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)

    return heat_map

def psnr_error(x, y, denorm = False, mu=-500, sigma=500):
    if denorm == False:
        x = denormalize(x,mu,sigma)
        y = denormalize(y,mu,sigma)
    if x.dim() == 2:
        h, w  = x.shape[0], x.shape[1]
        x = x.view(1,1,h,w)
        y = y.view(1,1,h,w)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    psnr_metric = PSNR(data_range=torch.max(x))
    psnr_metric.update((x, y))
    return psnr_metric.compute()

def ssim_error(x, y, denorm = False, mu=-500, sigma=500):
    if denorm == False:
        x = denormalize(x,mu,sigma)
        y = denormalize(y,mu,sigma)
    if x.dim() == 2:
        h, w  = x.shape[0], x.shape[1]
        x = x.view(1,1,h,w)
        y = y.view(1,1,h,w)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    ssim_metric = SSIM(data_range=torch.max(x))
    ssim_metric.update((x, y))
    return ssim_metric.compute()
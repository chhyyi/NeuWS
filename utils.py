import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import logging, os

import torch.nn.functional as F

from torch.fft import fftn, ifftn, rfftn, irfftn, fft2
from torch.fft import fftshift as fft_shift, ifftshift as ifft_shift

ang_to_unit = lambda x : ((x / np.pi) + 1) / 2

def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[-2] / 2.0), -2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[-1] / 2.0), -1)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[-2] / 2.0), -2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[-1] / 2.0), -1)
    return tensor_shifted

def ifft2c(arr):
    return ifftshift(ifftn(ifftshift(arr), norm="ortho"))

def fft2c(arr):
    return fftshift(fftn(fftshift(arr), norm="ortho"))


def compute_modulation(img, psf):
    f_img, otf = fft2c(img), fft2c(psf)
    f_res = (f_img * otf) / torch.abs(otf).max()
    res = ifft2c(f_res)
    
    return res.abs()
    

def align_images(recon, ref):
    """
    This implementation is desinged to extract relative shift with respect to 'ref'.
    To do so, the 'Cross-Correlation' in Fourier space is employed to find out the shift.
    Input shapes: [1, H, W]
    """
    recon_zeromean = recon - torch.mean(recon, dim=(1,2), keepdim=True)
    ref_zeromean = ref - torch.mean(ref, dim=(1,2), keepdim=True)

    recon_fft = fft2c(recon_zeromean)
    ref_fft = fft2c(ref_zeromean)

    cross_correlation = ifft2c(recon_fft * torch.conj(ref_fft))
    # cross_correlation = ifft2c(torch.conj(recon_fft) * ref_fft)
    cross_correlation = torch.abs(cross_correlation)

    _, idx = torch.max(cross_correlation.view(1, -1), 1)
    # max_row, max_col = idx // cross_correlation.shape[2], idx % cross_correlation.shape[2]
    max_row = torch.div(idx, cross_correlation.shape[2], rounding_mode='floor')
    max_col = torch.remainder(idx, cross_correlation.shape[2])

    _, rows, cols = cross_correlation.shape
    delM1 = (max_row.item() - rows // 2) % rows
    delM2 = (max_col.item() - cols // 2) % cols

    x_aligned = torch.roll(recon, shifts=(-delM1, -delM2), dims=(-2, -1))

    return x_aligned, delM1, delM2


def batchwise_align_images(recons, refs):
    """
    Applies align_images to each element in the batch.
    Input shapes: [B, 1, H, W]
    """
    B, _, H, W = recons.shape
    aligned_batch = []
    delM1_batch = torch.zeros(B, dtype=torch.long)
    delM2_batch = torch.zeros(B, dtype=torch.long)

    for i in range(B):
        aligned, delM1, delM2 = align_images(recons[i], refs[i])
        aligned_batch.append(aligned.unsqueeze(0))
        delM1_batch[i] = delM1
        delM2_batch[i] = delM2
        
    aligned_batch = torch.cat(aligned_batch, dim=0)
    return aligned_batch, delM1_batch, delM2_batch


def shift_correction(recon, ref, pad=False):
    B, C, H, W = recon.shape
    
    # Ensure inputs are float tensors
    recon = recon.float()
    ref = ref.float()
    
    # Remove mean
    recon_zeromean = recon - recon.mean(dim=(-2, -1), keepdim=True)
    ref_zeromean = ref - ref.mean(dim=(-2, -1), keepdim=True)
    
    if pad:
        # Compute FFT
        recon_fr = rfftn(recon_zeromean, dim=[-2, -1], s=[2*H, 2*W])
        ref_fr = rfftn(ref_zeromean, dim=[-2, -1], s=[2*H, 2*W])
    
        # Compute cross-correlation (with conjugate)
        cross_correlation_fr = complex_matmul(torch.conj(recon_fr), ref_fr)
        # cross_correlation_fr = complex_matmul(torch.conj(ref_fr), recon_fr)
    
        # Inverse FFT to get spatial cross-correlation
        cross_correlation = irfftn(cross_correlation_fr, dim=[-2, -1], s=[2*H, 2*W])
    
        # Find the maximum correlation for each image in the batch
        _, idx = torch.max(cross_correlation.view(B, -1), 1)
    
        # Calculate the shift
        max_row = torch.div(idx, (2*W), rounding_mode='floor')
        max_col = torch.remainder(idx, (2*W))
    
        # Calculate the required shift
        shift_row = torch.where(max_row > H, max_row - 2*H, max_row)
        shift_col = torch.where(max_col > W, max_col - 2*W, max_col)
    
        # Apply the shift
        grid_x = torch.arange(W, device=recon.device).repeat(H, 1)
        grid_y = torch.arange(H, device=recon.device).unsqueeze(1).repeat(1, W)
    
        grid_x = (grid_x.unsqueeze(0) - shift_col.view(B, 1, 1)) % W
        grid_y = (grid_y.unsqueeze(0) - shift_row.view(B, 1, 1)) % H
    
        grid = torch.stack((grid_x / (W - 1) * 2 - 1, grid_y / (H - 1) * 2 - 1), dim=-1)
    
        x_aligned = F.grid_sample(recon, grid.repeat(C, 1, 1, 1), mode='bilinear', padding_mode='border', align_corners=True)
        
        return x_aligned, shift_row, shift_col
        
    else:
        recon_fr = rfftn(recon_zeromean, dim=[-2, -1])
        ref_fr = rfftn(ref_zeromean, dim=[-2, -1])
        
        cross_correlation_fr = (recon_fr) * torch.conj(ref_fr)
        cross_correlation = irfftn(cross_correlation_fr, dim=[-2, -1], s=[-1, -1])
        cross_correlation = ifft_shift(cross_correlation, dim=[-2, -1])
        cross_correlation = torch.abs(cross_correlation)
        
        # x_aligned, delM1, delM2 = batchwise_align_images(recon, ref)
        
        # Find the maximum correlation for each image in the batch
        _, idx = torch.max(cross_correlation.view(B, -1), 1)
        
        B, _, H, W = cross_correlation.shape
        max_row = torch.div(idx, W, rounding_mode='floor')
        max_col = torch.remainder(idx, W)
    
        delM1 = (max_row - H // 2) % H
        delM2 = (max_col - W // 2) % W
    
        # Apply the shift using roll for each image in the batch
        x_aligned = []
        for i in range(B):
            x_aligned.append(torch.roll(recon[i], shifts=(-delM1[i].item(), -delM2[i].item()), dims=(-2, -1)))
        x_aligned = torch.stack(x_aligned)
    
        return x_aligned, delM1, delM2

    

# ============================================================================================================


# https://github.com/fkodom/fft-conv-pytorch

def complex_matmul(a, b, groups = 1):
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])

def fft_noPad_Conv2D(signal, kernel):
    signal_fr = rfftn(signal, dim=[-2, -1])
    kernel_fr = rfftn(kernel, dim=[-2, -1])

    output_fr = signal_fr * kernel_fr
    #output_fr = complex_matmul(signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=[-2, -1], s=[-1, -1])
    output = ifft_shift(output, dim=[-2, -1])

    return output

def fft_2xPad_Conv2D(signal, kernel, groups=1):
    size = signal.shape[-1]

    signal_fr = rfftn(signal, dim=[-2, -1], s=[2 * size, 2 * size])
    kernel_fr = rfftn(kernel, dim=[-2, -1], s=[2 * size, 2 * size])

    output_fr = complex_matmul(signal_fr, kernel_fr, groups)
    output = irfftn(output_fr, dim=[-2, -1], s=[-1, -1])
    s2 = size//2
    output = output[:, :, s2:-s2, s2:-s2]

    return output

def gen_point_spread_function(phs):
    psf = fft_shift(fft2(phs, norm="forward"), dim=[-2, -1]).abs() ** 2
    psf = psf / torch.sum(psf, dim=[-2, -1], keepdim=True)
    psf = psf.flip(-2).flip(-1)
    
    return psf

# ============================================================================================================

def set_logger(save_path, log_name):
    os.makedirs(save_path, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] - %(message)s')

    file_handler = logging.FileHandler(os.path.join(save_path, log_name))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def evaluate_iqa_metric(pred, ref, iqa_lists, device='cuda'):
    eval_lists = []
    
    pred = pred.expand(1, 1, -1, -1).to(device)
    ref = ref.expand(1, 1, -1, -1).to(device)
    
    for iqa in iqa_lists:
        metric = iqa(pred, ref).cpu().mean().item()
        eval_lists.append(metric)
        
    return eval_lists
        
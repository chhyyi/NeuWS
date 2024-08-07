# --- zernike aberrated image generation code ---
# Reimplemented version that does not requires matlab engine, 
# But presented results used 'generate_aberration_pair.py' on this directory
# copied from https://github.com/chhyyi/aberration_sbcho/blob/chyi/zernike_abrr.py
# requires aotools https://pypi.org/project/aotools/
# See if __name__=="__main__": for usage
# default output directory is "outputs/"

import aotools
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision as tv
import math
import pathlib

# ---functions based on the utils.py of [Choi et al., neural-holography](https://github.com/computational-imaging/neural-holography/) ---
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
    return ifftshift(torch.fft.ifftn(ifftshift(arr), norm="ortho"))

def fft2c(arr):
    return fftshift(torch.fft.fftn(fftshift(arr), norm="ortho"))

# --- other functions ---

def within_ctf(xl, yl, ctf_size=0.45):
    assert xl==yl, NotImplementedError
    x, y = np.meshgrid(np.linspace(-1., 1., xl), np.linspace(-1., 1., yl))
    is_in_circle = torch.tensor([[1 if i<=ctf_size else 0 for i in j] for j in np.power(np.sum((x**2, y**2), axis=0), 0.5)])
    return is_in_circle

def imshow_with_colorbar(img):
    ax = plt.subplot()
    im = ax.imshow(img)
    plt.colorbar(im)
    plt.show()

def zernike_abrr_phase(xl, yl, zern_modes=(3,20), ctf_size=0.45, amp_noise=None):
    """
    Generate aberration phase from combination of zernike modes with random coefficients.
    Implementation of random aberration MATLAB code I received from SB Cho, based on the [Analyzing LASIK OPtical Data using Zernike Functions](https://www.mathworks.com/matlabcentral/fileexchange/7687-zernike-polynomials)
    ctf_size : size of the CTF in units of entire image size. 
    assert xl==yl... As I could not recognized MATLAB implementation is valid for other cases....
    
    inputs:
    xl, yl : width/height of aberration phase

    abbreviations:
    * CTF : coherent transfer function
    * OTF : optical transfer function
    * PSF : point spread function
    """
    if amp_noise!=None:
        raise NotImplementedError
    assert xl==yl, NotImplementedError


    skip_modes = np.zeros(zern_modes[0])
    z_coeffs = np.random.rand(zern_modes[1]-zern_modes[0])*2.0-1.0
    z_coeffs = np.concatenate((skip_modes, z_coeffs))
    abrr_phase = torch.tensor(aotools.functions.zernike.phaseFromZernikes(z_coeffs, np.min((xl, yl))))
    return abrr_phase
    abrr_phase_complex = torch.exp(1j*abrr_phase)
    if ctf_size:
        abrr_phase_complex = torch.mul(is_in_circle, abrr_phase_complex)
    return abrr_phase_complex

def save_abrr_imgs(ref, pixelsize, num_output=30, path="outputs", save_preview=True):
    # make directory if not exists.
    pathlib.Path(path).mkdir(exist_ok=True)

    # Load image, resize
    ref = torch.tensor(plt.imread("../brain_data/low_pass_filtered_GT.tif"))
    resize = tv.transforms.Resize((pixelsize, pixelsize))
    ref = resize(ref[None,None,...])
    for i in range(num_output):
        abrr_phase = zernike_abrr_phase(pixelsize, pixelsize)

        # ctf
        random_phase_complex = torch.exp(1j*abrr_phase)[None, None, ...]
        is_in_circle = within_ctf(pixelsize, pixelsize, ctf_size=0.45)[None, None, ...]

        random_psf = torch.pow(torch.abs(ifft2c(torch.mul(is_in_circle, random_phase_complex))),2)
        random_otf = fft2c(random_psf)
    
        # save aberrated image
        f_ref = fft2c(ref)
        f_abb = torch.mul(f_ref, random_otf/torch.abs(random_otf).max())
        abb_result = ifft2c(f_abb)
        plt.imsave(f"{path}/output{i}.png", abb_result.real[0][0])

        # save preview
        if save_preview:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
            ax1.set_title("applied aberration")
            im = ax1.imshow(torch.angle(random_phase_complex[0][0]))
            fig.colorbar(im, ax=ax1)
            ax2.set_title(f"aberrated PSF {i}")
            im = ax2.imshow(random_psf[0][0])
            fig.colorbar(im, ax=ax2)
            ax3.set_title(f"aberrated images")
            im = ax3.imshow(abb_result.real[0][0])
            fig.colorbar(im, ax=ax3)
            fig.savefig(f"{path}/preview{i}.png")
        

if __name__ == "__main__":
    
    # # %% zernike phases preview
    # abrr_phase = zernike_abrr_phase(256, 256)
    # imshow_with_colorbar(abrr_phase)
    # # %% random_psf preview
    # random_phase_complex = torch.exp(1j*abrr_phase)[None, None, ...]
    # is_in_circle = within_ctf(256, 256, ctf_size=0.45)[None, None, ...]

    # random_psf = torch.abs(ifft2c(torch.mul(is_in_circle, random_phase_complex)))
    # imshow_with_colorbar(random_psf[0][0])
    # #%% preview random otf
    # random_otf = fft2c(random_psf)
    #imshow_with_colorbar(random_otf[0][0].real)
    
    # %% Load image, resize
    ref = torch.tensor(plt.imread("../brain_data/low_pass_filtered_GT.tif"))
    resize = tv.transforms.Resize((256, 256))
    ref = resize(ref[None,None,...])

    # # %% Show aberrated image
    # f_ref = fft2c(ref)
    # f_abb = torch.mul(f_ref, random_otf/torch.abs(random_otf).max())
    # abb_result = ifft2c(f_abb)
    # imshow_with_colorbar(abb_result.real[0][0])
    
    # %% save output and previews like matlab code (applied aberration, aberrated PSF and images)
    save_abrr_imgs(ref, 256, save_preview=True)

# %%

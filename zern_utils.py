import numpy as np
import math
import torch
import random

from aotools.functions import zernikeArray


def zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """
    Compute the radial component of Zernike polynomial.
    
    Args:
    n (int): Radial degree
    m (int): Azimuthal degree
    r (np.ndarray): Radial coordinate
    
    Returns:
    np.ndarray: Radial component of Zernike polynomial
    """
    if (n - m) % 2:
        return np.zeros_like(r)
    
    R = np.zeros_like(r)
    for k in range((n - m) // 2 + 1):
        R += ((-1)**k * math.factorial(n-k) / 
              (math.factorial(k) * math.factorial((n+m)//2 - k) * math.factorial((n-m)//2 - k))) * r**(n-2*k)
    return R

def zernike_function(p: int, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute Zernike polynomial for given parameters.
    
    Args:
    p (int): Zernike mode index
    r (np.ndarray): Radial coordinate
    theta (np.ndarray): Angular coordinate
    
    Returns:
    np.ndarray: Computed Zernike polynomial
    """
    n = math.ceil((-3 + math.sqrt(9 + 8*p)) / 2)
    m = 2*p - n*(n+2)
    
    R = zernike_radial(n, abs(m), r)
    
    if m > 0:   Z = R * np.cos(m * theta)
    elif m < 0: Z = R * np.sin(-m * theta)
    else:       Z = R
    
    return Z

def generate_zernike_mode(tpixel: int, NApixel: float, m: int, norm: bool = False) -> np.ndarray:
    """
    Generate a Zernike mode.
    
    Args:
    tpixel (int): Total number of pixels
    NApixel (float): Diameter of the NA in pixels
    m (int): Zernike mode index
    
    Returns:
    np.ndarray: Generated Zernike mode
    """
    x = np.linspace(-tpixel/NApixel, tpixel/NApixel, tpixel)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    mask = r <= 1
    z = np.zeros_like(X)
    z[mask] = zernike_function(m, r[mask], theta[mask])
    
    if norm:    # Noll Normalization
        rad = math.ceil((-3 + math.sqrt(9 + 8*m)) / 2)
        azm = 2*m - rad*(rad+2)
        norm_f = np.sqrt((2 * (rad + 1)) / (1 + (azm == 0)))
        z *= norm_f
    
    return z

def generate_zernike_basis(tpixel: int, NApixel: float, num_modes: int = 28):
    """
    Generate a set of Zernike basis functions.
    
    Args:
    tpixel (int): Total number of pixels
    NApixel (float): Diameter of the NA in pixels
    num_modes (int): Number of Zernike modes to generate (default: 28)
    
    Returns:
    np.ndarray: Generated Zernike basis functions
    """
    basis = np.stack([generate_zernike_mode(tpixel, NApixel, m, norm=True) for m in range(num_modes)])
    return torch.FloatTensor(basis)




def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    if target_shape is None:
        return field
    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2
    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field



def compute_zernike_basis(num_polynomials, field_res, mode='circle'):
    assert mode in ['circle', 'square', 'limit']
    
    if mode == 'square':
        zernike_diam = int(np.ceil(np.sqrt(field_res[0]**2 + field_res[1]**2)))
        zernike = zernikeArray(num_polynomials, zernike_diam)
        zernike = crop_image(zernike, field_res, pytorch=False)
        zernike = torch.FloatTensor(zernike)
        
    elif mode == 'circle':
        zernike = torch.FloatTensor(zernikeArray(num_polynomials, field_res[0]))
    
    else: # limit
        imsize = field_res[0]
        NA = imsize * 0.45
        zernike = torch.FloatTensor(generate_zernike_basis(imsize, NA, num_polynomials, norm=True))
        
    return zernike


def within_CTF(imsize, CTFsize):
    x, y = np.meshgrid(np.linspace(-1., 1., imsize), np.linspace(-1., 1., imsize))
    isin_CTF = torch.FloatTensor([[1 if i<=CTFsize else 0 for i in j] for j in np.power(np.sum((x**2, y**2), axis=0), 0.5)])

    return isin_CTF


def generate_modulation_patterns(imsize=256, CTFsize=0.45, num_modules=100, fix_seed=None, module_coeff_std=(np.pi / 2.), mode='circle'):
    assert mode in ['circle', 'square', 'limit']
    
    in_CTF = within_CTF(imsize, CTFsize)

    modules = []
    if fix_seed is not None:
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)

    NA = imsize * CTFsize

    for _ in range(num_modules):
        abrr_pattern = torch.zeros(imsize, imsize)

        for _ in range(20):
            coeff = (1 - 2 * torch.rand(1)) * module_coeff_std
            zern_order = random.randint(3, 21)

            abrr_pattern += coeff * torch.FloatTensor(generate_zernike_mode(imsize, NA, zern_order, norm=True))

        # module = torch.angle(in_CTF * torch.exp(1j * abrr_pattern))
        modules.append(abrr_pattern.unsqueeze(0))

    return modules
# -*- coding: utf-8 -*-
import numpy as np
from scipy.fft import fftshift, fft2

def fft2c(x):
    
    res = 1/np.sqrt(np.prod(x.shape)) * fftshift(fft2(fftshift(x)))

    return res
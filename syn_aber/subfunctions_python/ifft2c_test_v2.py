# -*- coding: utf-8 -*-

import numpy as np
from scipy.fft import ifftshift, ifft2
#from numpy.fft import ifftshift, ifft2

def ifft2c(x):
    
    #res = np.sqrt(x.size) * (ifft2(ifftshift(x)))         # OTFphase가 이상하게 나옴
    res = np.sqrt(x.size) * ifftshift(ifft2(ifftshift(x)))
  
    
    return res

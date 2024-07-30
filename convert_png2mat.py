import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--png_pth", type=str, required=True, help = "path of png file to convert")
parser.add_argument("--var_name", type=str, required=True, help="name of array of image")

args = parser.parse_args()

## arguments
png_pth = args.png_pth
variable_name = args.var_name


## ----
png_pth = Path(png_pth)
mat_pth = png_pth.with_suffix(".mat")

img = plt.imread(png_pth)
print(f"img.max(): {img.max()} img.min(): {img.min()}, considering it is clipped in range [0, 1.0]... (I guess pyplot read uint8 images in that range)")
img = img*2.0*np.pi - np.pi
if len(np.shape(img))==4:
    img = img.squeeze(0)
if len(np.shape(img))==3:
    img = img.squeeze(0)

sio.savemat(mat_pth, {variable_name: img})
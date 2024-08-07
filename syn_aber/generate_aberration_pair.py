# -*- coding: utf-8 -*-
# Zernike polynomials (https://www.mathworks.com/matlabcentral/fileexchange/7687-zernike-polynomials)
# Python code to create aberration images using MATLAB's Zernike polynomials function.

import numpy as np
import matplotlib.pyplot as plt
import natsort
from PIL import Image
import tifffile as tiff
import os
import pickle

from subfunctions_python.ifft2c_test_v2 import ifft2c
from subfunctions_python.fft2c_test_v2 import fft2c


# CTF: coherent trasnfer function, 
# PSF: point spread function, 
# OTF: optical transfer function, 
# MTF: modulation transfer function
    



# -----------------------------------------------------------------------------
#                                load Images
# -----------------------------------------------------------------------------
directory_path = 'GroundTruth/path'

#image_paths = os.listdir(directory_path)
# load only tif files
image_paths = [file for file in os.listdir(directory_path) if file.endswith('.tif')]
# Image sorted
image_paths = natsort.natsorted(image_paths)

ref_images = []
for path in image_paths:
    image = tiff.imread(os.path.join(directory_path, path))
    ref_images.append(np.array(image))
ref_images = np.array(ref_images)


# -----------------------------------------------------------------------------
#                             variable value set
# -----------------------------------------------------------------------------
c1 = ref_images.shape[1]
#c2 = ref_images.shape[0]
c2 = ref_images.shape[2]

# CTFsize < 0.5
CTFsize = 0.45  # size of the CTF in units of entire image size
L1 = c2          # width of image
L2 = c1
Y = (np.linspace(-1, 1, L2)).reshape(1,-1)     # define coordinates for zernike polynomial generation
X = (np.linspace(-1, 1, L1)).reshape(1,-1)
x,y = np.meshgrid(X,Y)                      # define coordinates for zernike polynomial generation


theta, r = np.arctan2(y,x), np.hypot(x,y)   # use polar coordinates to generate Zernike polynomials


is_in_circle2 = (r <= CTFsize).astype(float)

ideal_psf = np.abs(ifft2c(is_in_circle2))**2   # ideal PSF for the defined CTF
OTF_ideal = fft2c(ideal_psf)                   # ideal OTF for the defined CTF : complex로 바뀜


'''
plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(np.abs(OTF_ideal))   # cmap='gray'
plt.colorbar(); plt.title('OTF Magnitude')
plt.subplot(132); 
plt.imshow(np.angle(OTF_ideal)) #hsv
plt.colorbar(); plt.title('OTF Phase'); plt.clim(0,)
plt.subplot(133)
plt.imshow(np.real(OTF_ideal))
plt.colorbar(); plt.title('OTF Real')
plt.show()
'''



#####################################################

C = len(ref_images)

D = 20                  # the number of zernike basis modes to be combined
zern_range = 20          # max range of zernike basisorder
abb_degree= np.pi/1.5   # max amplitude range of each zernike mode added

psf_data = np.zeros((c1,c2,C))
images_data = np.zeros((c1,c2,C))


pixelnumber = c1  # matrix size
pixelnumberaperture = c1 * CTFsize  # circle size of zernike basis


# -----------------------------------------------------------------------------
#                          import MATLAB function
# -----------------------------------------------------------------------------
import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(r'subfunctions', nargout=0)
eng.eval('gzn2', nargout=0)

# Convert Python variables to MATLAB doubles
pixelnumber = matlab.double([pixelnumber])
pixelnumberaperture = matlab.double([pixelnumberaperture])



# -----------------------------------------------------------------------------
#                       information data save path
# -----------------------------------------------------------------------------
np.random.seed(42)

save_base_directory = 'base/directory'

aberration_directory = os.path.join(save_base_directory, 'channel10_Aberrated')
phase_directory = os.path.join(save_base_directory, 'channel10_Amplitude')
#psf_directory = os.path.join(save_base_directory, 'channel10_PSF')
zernike_coefficient_directory = os.path.join(save_base_directory, 'channel10_coefficients')
collect_coeff_directory = os.path.join(save_base_directory, 'collect_poly_coeff')
collect_Zern_order_directory = os.path.join(save_base_directory, 'collect_Zern_order')
random_phase_directory = os.path.join(save_base_directory, 'random_phase')


# -----------------------------------------------------------------------------
#                     generate synthetic aberration data
# -----------------------------------------------------------------------------
#plt.ion()
for ci in range(C):
    for j in range(10):
        collect_Zern_order = []
        collect_poly_coeff = []
        combined_amplitudes = np.zeros((zern_range, C))
        
        random_aberration_complex = np.ones((c1,c2))
        
       # Iterative combination of Zernike basis
        
        for di in range(D):
            poly_coeff2 = (1- 2*np.random.rand()) * (abb_degree)  # randomly generated amplitude
            collect_poly_coeff.append(poly_coeff2)
            
            Zern_order2 = np.random.randint(3, zern_range+1)   # randomly selected zernike basis, do not use piston, tip, tilt
            
            
            # Convert Zern_order2 to MATLAB-compatible data type
            matlab_zern_order = matlab.double([float(Zern_order2)])
            
            # Call the gzn2 function
            W1_matlab = eng.gzn2(pixelnumber, pixelnumberaperture, matlab_zern_order)
            
            ####################################
            # Convert the output back to a NumPy array and resize to match the shape of random_aberration_complex
            W1 = np.resize(np.array(W1_matlab), random_aberration_complex.shape)
            ###################################
            
            # Convert the output back to a NumPy array
            #W1 = np.array(W1_matlab)
            W1 = poly_coeff2 * W1
            
            
        
            #W1 = np.array(eng.gzn2(pixelnumber, pixelnumberaperture, matlab.double([Zern_order2[1]])))
            
            random_aberration_complex = random_aberration_complex*np.exp(1j*W1) 
         
        random_phase = random_aberration_complex
            
# =============================================================================
#         # ---------------------------------------------------------------------
#         #       Generate fixed aberration + random aberration dataset
#         # ---------------------------------------------------------------------         
#
#         # Load Fixed_random_phase values for make fixed_random_dataset)
#         with open(save_base_directory +f"/Fixed/random_phase/random_phase{ci}_{0}.pickle","rb") as fr:
#             fixed_random_phase = pickle.load(fr)
#                
#         random_phase = random_aberration_complex * fixed_random_phase    # generated random aberration in phase
#         
#         
#         # save random_phase file as pickle (for make fixed+random dataset)
#         with open(random_phase_directory +f"/random_phase{ci}_{j}.pickle","wb") as fw:
#             pickle.dump(random_phase, fw)
# =============================================================================
        
        
        random_psf = np.abs(ifft2c(is_in_circle2*random_phase))**2  # generated random aberrated PSF
        OTF_rand = fft2c(random_psf)    # generated randomly aberrated OTF
    
        psf_data[..., ci] = random_psf   # randomaberrated PSFs
        
        ###### make randomly aberrated images ######
        f_ref = fft2c(ref_images[ci, ...])
        #f_ref = fft2c(ref_images)               # one GT image
        f_abb = f_ref*OTF_rand/np.abs(np.max(np.max(OTF_rand)))
        abb_result = ifft2c(f_abb)
        
        images_data[..., ci] = np.abs(abb_result)    # random aberrated images
        
        
        # ---------------------------------------------------------------------
        #                     save the datas
        # ---------------------------------------------------------------------
        aberrated_filename = os.path.join(aberration_directory, f'{ci}_{j}.tif')
        
        # Extract the magnitude of the complex values before saving
        abb_result_magnitude = images_data[..., ci]
                
        # Convert the magnitude to a PIL Image and save
        #PIL_abb_result = Image.fromarray(np.float64(abb_result_magnitude))
        PIL_abb_result = Image.fromarray(np.uint8(abb_result_magnitude))
        PIL_abb_result.save(aberrated_filename)
        
        
        ######  Save each image with a unique aberration images  ######
        phase_filename = os.path.join(phase_directory, f'{ci}_{j}.tif')
        
        angle_random_phase = np.angle(random_phase)
        PIL_random_phase = Image.fromarray(np.float64(angle_random_phase))
        #PIL_random_phase = Image.fromarray(np.uint8(angle_random_phase))
        PIL_random_phase.save(phase_filename)
        
        
# =============================================================================
#         ## save each psf_data
#         psf_data_filename = os.path.join(psf_directory, f'{ci}_{j}.tif')
#         psf_data_result = Image.fromarray(np.float64(psf_data[...,ci]))
#         psf_data_result.save(psf_data_filename)
# =============================================================================
        
        
        ## save each Zernike coefficients
        #zernike_coeff_filename = os.path.join(psf_directory, f'{ci}_{j}.tif')
        
        for k in range(1,zern_range+1):
            indices = np.where(np.array(collect_Zern_order) == k)[0]  # Find indices where Zernike index equals k
            collect_poly_coeff = np.array(collect_poly_coeff)
            if len(indices) > 0:
                combined_amplitudes[k - 1, :] = np.sum(collect_poly_coeff[indices])  # Sum up the amplitudes
        
        # Plotting combined Zernike polynomial amplitudes
        plt.figure()
        plt.stem(np.arange(1, zern_range + 1), combined_amplitudes[:, ci], use_line_collection=True)
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        plt.xlabel('Zernike Mode Number')
        plt.ylabel('Amplitude')
        plt.title('Combined Zernike Polynomial coefficients')
        plt.tight_layout()
        plt.savefig(zernike_coefficient_directory + f'/{ci}_{j}.png')
        plt.show()
        
        
        # Save collect_Zern_order as pickle (for make fixed+random dataset)
        with open(collect_Zern_order_directory +f"/{ci}_{j}.pickle","wb") as fw:
            pickle.dump(collect_Zern_order, fw)
        
        with open(collect_coeff_directory +f"/{ci}_{j}.pickle","wb") as fw:
            pickle.dump(collect_poly_coeff, fw)
        
# =============================================================================        
#       # ---------------------------------------------------------------------
#       #            display each aberrated images by 0.1seconds
#       # ---------------------------------------------------------------------
#
#     plt.figure(8, figsize=(12,3))
#     plt.subplot(141) # cmap='jet'
#     plt.imshow(np.angle(random_phase), cmap='gray');plt.axis('image'); plt.title('Applied Aberration'); plt.colorbar()#, plt.clim(-3,3)
#     
#     plt.subplot(142)
#     plt.imshow(images_data[..., ci], cmap='gray'); plt.axis('image'); plt.title('Aberrated Images'); plt.colorbar()
#         
#     plt.subplot(143)
#     plt.imshow(ref_images[ci, ...], cmap='gray'); plt.axis('image'); plt.title('Clean Images'); plt.colorbar()
#     
#     #plt.subplot(144)
#     #plt.imshow(psf_data[..., ci], cmap='gray'); plt.axis('image'); plt.title(f'Aberrated PSF {ci+1}'); plt.colorbar()#, plt.clim(0,10)
#     
#     plt.show()
#     
#     plt.draw()       # Update the figure
#     plt.pause(0.1)   # Pause for a short time (e.g., 0.1 seconds) to allow real-time update
#     
#     
#     #plt.clf()
#     
# plt.ioff()  # Disable interactive mode
# =============================================================================
#plt.tight_layout()
#plt.show()  # Show the final figure
print('done')
eng.quit()




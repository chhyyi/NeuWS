# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
import natsort
import tifffile
import random
import aotools

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *
from zern_utils import *
from cancel_defocus import cancel_defocus

class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, im_prefix='SLM_raw', slm_prefix='SLM_sim', num=100, max_intensity=0, zero_freq=-1):
        self.data_dir = data_dir
        self.zero_freq = zero_freq
        a_slm = np.ones((144, 256))
        a_slm = np.lib.pad(a_slm, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
        self.a_slm = torch.from_numpy(a_slm).type(torch.float)
        self.max_intensity = max_intensity
        self.num = num
        self.im_prefix, self.slm_prefix = im_prefix, slm_prefix
        self.load_in_cache()
        self.num = len(self.xs)
        print(f'Training with {self.num} frames.')

    def load_in_cache(self):
        x_list, y_list = [], []
        for idx in range(self.num):
            img_name = f'{self.data_dir}/{self.im_prefix}{idx+1}.mat'
            mat_name = f'{self.data_dir}/{self.slm_prefix}{idx+1}.mat'

            try:
                p_SLM = sio.loadmat(f'{mat_name}')
                p_SLM = p_SLM['proj_sim']
                p_SLM = np.lib.pad(p_SLM, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
                p_SLM_train = torch.FloatTensor(p_SLM).unsqueeze(0)

                if self.zero_freq > 0 and idx % self.zero_freq == 0:
                    p_SLM_train = torch.zeros_like(p_SLM_train)
                    img_name = f'{self.data_dir}/../Zero/{self.im_prefix}{idx+1}.mat'
                    print(f'#{idx} uses zero SLM')

                x_train = self.a_slm * torch.exp(1j * -p_SLM_train)
                ims = sio.loadmat(f'{img_name}')
                y_train = ims['imsdata']

                if np.max(y_train) > self.max_intensity:
                    self.max_intensity = np.max(y_train)

                y_train = torch.FloatTensor(y_train)
                x_list.append(x_train); y_list.append(y_train)

            except Exception as e:
                print(f'{e}')
                continue
        y_list = [y / self.max_intensity for y in y_list]
        self.xs, self.ys = x_list, y_list

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], idx
    

    
class BrainDataset(Dataset):
    def __init__(self, target_dir, imsize=256, CTFsize = 0.45, num_modules=100, module_coeff_std = math.pi,
                 random_abrr=False, fix_module=None, cohereng_avg = False,
                 abrr_order=7, abrr_std_tuple=(3, 4), rand_std_tuple = (1.5, 1.5)):
        
        self.target_dir = target_dir
        self.target = torch.FloatTensor(tifffile.imread(self.target_dir)) / 255.
        
        self.num_modules = num_modules
        self.imsize = imsize
        self.CTFsize = CTFsize
        self.module_coeff_std = module_coeff_std
        
        self.random_abrr = random_abrr
        self.fix_module = fix_module
        self.coherent_avg = cohereng_avg
        
        self.abrr_order = abrr_order
        self.abrr_std_tuple = abrr_std_tuple
        self.rand_std_tuple = rand_std_tuple
        
        def within_CTF(imsize=self.imsize, CTFsize=self.CTFsize):
            x, y = np.meshgrid(np.linspace(-1., 1., imsize), np.linspace(-1., 1., imsize))
            isin_CTF = torch.FloatTensor([[1 if i<=CTFsize else 0 for i in j] for j in np.power(np.sum((x**2, y**2), axis=0), 0.5)])
    
            return isin_CTF
        
        def gen_modulation_patterns(imsize=self.imsize, CTFsize=self.CTFsize, num_modules=self.num_modules,
                                    fix_seed=None, module_coeff_std=self.module_coeff_std):
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
                    
                modules.append(abrr_pattern.unsqueeze(0))
                
            return modules
        
        def gen_fixed_synthetic_aberration(imsize=self.imsize, CTFsize=self.CTFsize, abrr_order=self.abrr_order,
                                           zern_std_tuple = self.abrr_std_tuple, rand_std_tuple = self.rand_std_tuple,
                                           random_abrr=False):
            torch.manual_seed(42)
            random.seed(42)
            
            zern_std_low, zern_std_high = zern_std_tuple[0], zern_std_tuple[1]
            rand_std_low, rand_std_high = rand_std_tuple[0], rand_std_tuple[1]
            num_polys = abrr_order * (abrr_order + 1) // 2; NA = imsize * CTFsize
            
            zern_std = torch.FloatTensor(1).uniform_(zern_std_low, zern_std_high)
            rand_std = torch.FloatTensor(1).uniform_(rand_std_low, rand_std_high)
            
            syn_abrr = torch.zeros(1, imsize, imsize)
            for _ in range(num_polys):
                coeff = (1 - 2 * torch.rand(1)) * zern_std
                zern_order = random.randint(3, num_polys)
                syn_abrr += coeff * torch.FloatTensor(generate_zernike_mode(imsize, NA, zern_order, norm=True))
                
            SLM_phs = syn_abrr
            SLM = torch.exp(1j * SLM_phs)
            
            if random_abrr:
                mean, std = 0, rand_std
                
                rand_phs = torch.randn_like(torch.abs(SLM)) * std + mean
                rand_phs = transforms.Resize((SLM.size()[-2]//4, SLM.size()[-1]//4), interpolation=transforms.InterpolationMode.NEAREST)(rand_phs)
                rand_phs = transforms.Resize((SLM.size()[-2], SLM.size()[-1]), interpolation=transforms.InterpolationMode.NEAREST)(rand_phs)
                
                SLM_phs = self.within_CTF * rand_phs
                SLM = torch.exp(1j * SLM_phs)
                
            return SLM, SLM_phs
        
        self.within_CTF = within_CTF(self.imsize, self.CTFsize)
        self.ideal_psf = (torch.abs(ifft2c(self.within_CTF)) ** 2).unsqueeze(0)
        self.ideal_otf = fft2c(self.ideal_psf)
        
        self.phase_list = gen_modulation_patterns(imsize=self.imsize, num_modules=self.num_modules, CTFsize=self.CTFsize, fix_seed=self.fix_module)
        self.fixed_syn_abrr, self.fixed_syn_phs = gen_fixed_synthetic_aberration(imsize=self.imsize, abrr_order=self.abrr_order, random_abrr=self.random_abrr)
        self.fixed_syn_abrr = self.within_CTF * self.fixed_syn_abrr
        self.abrr_psf = torch.abs(ifft2c(self.within_CTF * self.fixed_syn_abrr)) ** 2
        self.modulate()
        
        print(f'Training with {len(self.phase_list)} frames.')
        
        
    def __len__(self):
        return len(self.phase_list)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
        
    def modulate(self):
        obs_list, mod_list = [], []
        target = self.target.unsqueeze(0)
        self.uncorr = self.fft_conv(target, self.abrr_psf)
        
        for phs in self.phase_list:
            mod = torch.exp(1j * phs)
            syn = self.fixed_syn_abrr
            
            phs_comb = mod * syn
            psf = torch.abs(ifft2c(self.within_CTF * phs_comb)) ** 2
            if self.coherent_avg:
                psf, _, _ = align_images(psf, self.ideal_psf)
            res = self.fft_conv(target, psf)
            
            obs_list.append(res[0])
            mod_list.append(self.within_CTF * mod)
            
        self.conj_abrr = torch.conj(self.fixed_syn_abrr)
        self.corr_psf = torch.abs(ifft2c(self.within_CTF * (self.conj_abrr * self.fixed_syn_abrr))) ** 2
        self.ideal_target = self.fft_conv(self.target, self.corr_psf).squeeze()
            
        self.xs = mod_list
        self.ys = obs_list
            
    def fft_conv(self, ref, psf):
        f_ref, otf = fft2c(ref), fft2c(psf)
        f_res = (f_ref * otf) / torch.abs(otf).max()
        
        return ifft2c(f_res).abs()
    
    
    def update_new_aberration(self, new_abrr, new_mod_coeff_std):
        self.fixed_syn_abrr = new_abrr
        self.abrr_psf = torch.abs(ifft2c(self.within_CTF * self.fixed_syn_abrr)) ** 2
        # self.phase_list = generate_modulation_patterns(fix_seed=self.fix_module, module_coeff_std=new_mod_coeff_std)
        
        self.modulate()
        
    
    
class RealDataset(Dataset):
    def __init__(self, target_dir, gt_dir, depth=100, imsize=256, CTFsize=1., num_modules=100, fix_module=42):
        
        self.target_dir = target_dir
        self.gt_dir = gt_dir
        self.source_img = tifffile.imread(self.target_dir)
        self.normal_img = [torch.FloatTensor((self.source_img[i] - self.source_img[i].min())/(self.source_img[i].max() - self.source_img[i].min())) \
                            for i in range(self.source_img.shape[0])]    # min-max normalization for each image
        
        self.sys_corr  = self.normal_img[0]
        self.full_corr = np.array(Image.open(gt_dir))#gt. not correction 
        
        def within_CTF(imsize=imsize, CTFsize=CTFsize):
            x, y = np.meshgrid(np.linspace(-1., 1., imsize), np.linspace(-1., 1., imsize))
            isin_CTF = torch.FloatTensor([[1 if i<=CTFsize else 0 for i in j] for j in np.power(np.sum((x**2, y**2), axis=0), 0.5)])
    
            return isin_CTF
        
        def gen_modulation_modes(imsize=imsize, CTFsize=CTFsize, module_coeff_std=(np.pi/4.), fix_module=fix_module):
            # in_CTF = within_CTF(imsize, CTFsize)

            modules = []
            if fix_module is not None:
                random.seed(fix_module)
                torch.manual_seed(fix_module)

            NA = imsize * CTFsize

            for _ in range(num_modules):
                abrr_modes = torch.zeros(imsize, imsize)

                for _ in range(20):
                    coeff = (1 - 2 * torch.rand(1)) * module_coeff_std
                    zern_order = random.randint(3, 21)

                    abrr_modes += coeff * torch.FloatTensor(generate_zernike_mode(imsize, NA, zern_order, norm=True))

                # module = torch.angle(in_CTF * torch.exp(1j * abrr_pattern))
                modules.append(abrr_modes.unsqueeze(0))

            return modules
        
        self.isin_CTF = within_CTF()
        self.module_modes = gen_modulation_modes()
        self.load_train_data()
        
        
    def __len__(self):
        return len(self.module_modes)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
        
    def load_train_data(self):
        phs_list = []
        for i in range(len(self.module_modes)):
            mod = self.module_modes[i]
            phs_list.append(self.isin_CTF * torch.exp(1j * mod))
        
        self.xs = phs_list[-8:] + phs_list[:-8]
        self.ys = self.normal_img[1:]

        
class RealDataset_ModFromTif(RealDataset):
    def __init__(self, target_dir, gt_dir, mod_tif_dir, depth=100, imsize=256, CTFsize=1., num_modules=100, fix_module=42, compensate_defocus=True, rot_input=0.0):
        super().__init__(target_dir, gt_dir, depth=100, imsize=256, CTFsize=1., num_modules=100, fix_module=42)
        self.imsize= imsize
        self.mod_tif_dir = Path(mod_tif_dir)
        self.mod_base_name = self.mod_tif_dir.stem
        self.compensate_defocus = compensate_defocus
        self.rot_abrr = rot_input
        self.xs = [self.mod_from_tif(f"{self.mod_base_name}_{str(i)}.tif") for i in range(1, len(self.module_modes)+1)]
    def mod_from_tif(self, pth):
        mod = tifffile.imread(self.mod_tif_dir.joinpath(pth))
        mod = torch.from_numpy(mod).type(torch.float)
        mod = transforms.Resize(self.imsize)(mod[None,...])
        if self.compensate_defocus:
            mod = cancel_defocus(mod)
        if self.rot_abrr:
            mod = transforms.functional.rotate(mod, self.rot_abrr)
        return self.isin_CTF * torch.exp(1j * mod)
        
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
        
# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import torch
import numpy as np
import scipy.io as sio
import tifffile
from torchvision import transforms

class TifDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, im_tif_pth=None, mod_tif_pth=None, num=100, max_intensity=0, zero_freq=-1, circular_mask=0, phase_rot = 0):
        self.obs_pth = im_tif_pth
        self.mod_pth = mod_tif_pth

        # load observation (image)
        obs = tifffile.imread(self.obs_pth) #observed image with random aberration
        assert np.shape(obs)==(102, 256, 256), "Not Implemented..."
        self.obs = torch.FloatTensor(obs[2:].astype(np.float32)) #except two 
        
        #self.system_corrected = self.obs[0].astype(dtype)/65535.0
        #self.impact_corrected = self.obs[1].astype(dtype)/65535.0

        # load modulations
        def within_CTF(imsize=960, CTFsize=1.0):
            x, y = np.meshgrid(np.linspace(-1., 1., imsize), np.linspace(-1., 1., imsize))
            isin_CTF = torch.FloatTensor([[1 if i<=CTFsize else 0 for i in j] for j in np.power(np.sum((x**2, y**2), axis=0), 0.5)])
            return isin_CTF
        mod = torch.FloatTensor(tifffile.imread(self.mod_pth)) #It is ~ 360MB.. let's just calc at init.
        assert np.shape(mod)==(100, 960, 960), "Not implemented..."
        if circular_mask:
            mod = within_CTF(CTFsize=circular_mask)*mod

        mod = transforms.Resize(256)(mod)
        
        #pad 192 to 256
        #mod = np.lib.pad(mod, ((0,0),((256 - 192) // 2, (256 - 192) // 2), ((256 - 192) // 2, (256 - 192) // 2)), 'constant', constant_values=(0, 0))

        self.mod = transforms.functional.rotate(torch.FloatTensor(mod), phase_rot)
        
        # ###########################
        self.data_dir = data_dir
        self.zero_freq = zero_freq
        
        
        a_slm = np.ones((256, 256))
        
        #pad 192 to 256
        #a_slm = np.ones((192, 192))
        #a_slm = np.lib.pad(a_slm, (((256 - 192) // 2, (256 - 192) // 2), ((256 - 192) // 2, (256 - 192) // 2)), 'constant', constant_values=(0, 0))

        #a_slm = np.ones((144, 256))
        #a_slm = np.lib.pad(a_slm, (((256 - 144) // 2, (256 - 144) // 2), (0, 0)), 'constant', constant_values=(0, 0))
        self.a_slm = torch.from_numpy(a_slm).type(torch.float)
        self.max_intensity = max_intensity
        self.num = num
        self.load_in_cache()
        self.num = len(self.mod)
        print(f'Training with {self.num} frames.')

    
    def load_in_cache(self):
        x_list, y_list = [], []
        for idx in range(self.num):
            #img_name = f'{self.data_dir}/{self.im_prefix}{idx+1}.mat'
            #mat_name = f'{self.data_dir}/{self.slm_prefix}{idx+1}.mat'

            #try:
            p_SLM = self.mod[idx]

            p_SLM_train = torch.FloatTensor(p_SLM).unsqueeze(0)

            #---hard coded--- rescale
            p_SLM_train = (p_SLM_train-p_SLM_train.min()+0.1)/(p_SLM_train.max()-p_SLM_train.min())*41000+3000

            x_train = self.a_slm * torch.exp(1j * -1.0*p_SLM_train) #slm res mask * torch.exp(1j * phase in rad), phase in rad: unwrapped (1, 256, 256)
            y_train = self.obs[idx]

            if y_train.max() > self.max_intensity:
                self.max_intensity = y_train.max()

            y_train = torch.FloatTensor(y_train) # (256, 256)
            x_list.append(x_train); y_list.append(y_train)

            #except Exception as e:
             #   print(f'{e}')
               # continue
        y_list = [y / self.max_intensity for y in y_list]
        self.xs, self.ys = x_list, y_list

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], idx
    
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

                x_train = self.a_slm * torch.exp(1j * -1.0*p_SLM_train)
                ims = sio.loadmat(f'{img_name}')
                y_train = ims['imsdata']

                if np.max(y_train) > self.max_intensity:
                    self.max_intensity = np.max(y_train)

                y_train = torch.FloatTensor(y_train) # uint16 -> float
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



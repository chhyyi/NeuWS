import os, time, math, imageio, tqdm, argparse, pyiqa
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import time
from datetime import datetime, timedelta

import torch
print(f"Using PyTorch Version: {torch.__version__}")
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

import torch.nn.functional as F
from torch.fft import fft2, fftshift
from networks import *
from utils import *
from data import RealDataset, RealDataset_ModFromTif
import tifffile

DEVICE = 'cuda'
# #python main_static.py --root_dir ./brain_data --data_dir low_pass_filtered_GT.tif --num_t 100 --phs_layer 8 --num_epochs 1000 --coeff_std_factor 2 --batch_size 32 \
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--data_dir', default=r'datasets\dataset8.02\dataset2\najifixedaberration.tiff', type=str)
    parser.add_argument('--gt_dir', default=r'datasets\dataset8.02\dataset2\GTimage18umdepth20avf15%pw.tif',type=str)
    parser.add_argument('--mod_tif_dir', default=r'datasets\modulation_swcho0725_defocus_canceled\2',type=str)
    parser.add_argument('--vis_dir', default=r'vis/hw_data0802_dset2_abrr2/naji-fixedabrr_rot90',type=str)
    parser.add_argument('--rot_input', default=90, type=float)


    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_t', default=100, type=int)
    parser.add_argument('--depth', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--init_lr', default=1e-3, type=float)
    parser.add_argument('--final_lr', default=1e-3, type=float)
    parser.add_argument('--silence_tqdm', action='store_true')
    parser.add_argument('--save_per_frame', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--phs_layers', default=8, type=int)
    parser.add_argument('--log_name', default='result.log', type=str)
    
    args = parser.parse_args()
    PSF_size = args.width
    args.vis_freq = args.vis_freq // (args.batch_size // 8)
    
    data_dir = f'{args.root_dir}/{args.data_dir}'
    vis_dir = f'{args.root_dir}/{args.vis_dir}'
    gt_dir = f'{args.root_dir}/{args.gt_dir}'
    os.makedirs(f'{args.root_dir}/vis', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(f'{vis_dir}/final', exist_ok=True)
    print(f'Saving output at: {vis_dir}')
    
    logger = set_logger(vis_dir, args.log_name)
    logger.info(args)
    
    # IQA Metrics
    DISTS = pyiqa.create_metric('dists', as_loss=False, device=DEVICE)
    LPIPS = pyiqa.create_metric('lpips', as_loss=False, device=DEVICE)
    PSNR  = pyiqa.create_metric('psnr', as_loss=False, device=DEVICE)
    SSIM  = pyiqa.create_metric('ssim', as_loss=False, device=DEVICE)
    iqa_lists = [DISTS, LPIPS, PSNR, SSIM]

    #dset = RealDataset(target_dir=data_dir, gt_dir=gt_dir, depth=args.depth)
    dset = RealDataset_ModFromTif(target_dir=data_dir, gt_dir=gt_dir, mod_tif_dir=args.mod_tif_dir, depth=args.depth, compensate_defocus=False, rot_input=args.rot_input)
    x_batches = torch.cat(dset.xs, axis=0).unsqueeze(1).to(DEVICE)
    y_batches = torch.stack(dset.ys, axis=0).to(DEVICE)
    
    isin_CTF = dset.isin_CTF.to(DEVICE)[None, None, ...]
    CTF = isin_CTF.detach().cpu().squeeze()
    target = torch.tensor(dset.full_corr, dtype=torch.float)[None,None,...]/65535.0
    
    net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=True, use_CTF=isin_CTF,
                           coherent_avg=False)
    net = net.to(DEVICE)
    
    im_opt = torch.optim.Adam(net.g_im.parameters(), lr=args.init_lr)
    ph_opt = torch.optim.Adam(net.g_g.parameters(), lr=args.init_lr)
    im_sche = torch.optim.lr_scheduler.CosineAnnealingLR(im_opt, T_max = args.num_epochs, eta_min=args.final_lr)
    ph_sche = torch.optim.lr_scheduler.CosineAnnealingLR(ph_opt, T_max = args.num_epochs, eta_min=args.final_lr)

    total_it = 0
    t = tqdm.trange(args.num_epochs, disable=args.silence_tqdm)
    
    t0 = time.time()
    logger.info("="*50)
    logger.info(f"Time : {datetime.now()}\n")
    for epoch in t:
        idxs = torch.randperm(len(dset)).long().to(DEVICE)
        for it in range(0, len(dset), args.batch_size):
            idx = idxs[it:it+args.batch_size]
            x_batch, y_batch = x_batches[idx], y_batches[idx]
            cur_t = (idx / (args.num_t - 1)) - 0.5
            im_opt.zero_grad();  ph_opt.zero_grad()

            y, _kernel, sim_g, sim_phs, I_est = net(x_batch, cur_t)
            
            mse_loss = F.mse_loss(y, y_batch)

            loss = mse_loss
            loss.backward()

            ph_opt.step()
            im_opt.step()

            t.set_postfix(MSE=f'{mse_loss.item():.4e}')
            
            if args.vis_freq > 0 and (total_it % args.vis_freq) == 0:
                y, _kernel, sim_g, sim_phs, I_est = net(x_batch, torch.zeros_like(cur_t) - 0.5)
                sim_angle = torch.angle(torch.exp(1j * sim_phs[0]).detach().cpu().squeeze())
                
                # =================== Compute Optical Correction ===================
                sim_conj = torch.conj(sim_g[0].detach().cpu().squeeze())
                
                if I_est.shape[0] > 1:
                    I_est = I_est[0:1]
                I_est = torch.clamp(I_est, 0, 1)
                
                obj_est = evaluate_iqa_metric(I_est, target, iqa_lists)
                est_dists, est_lpips, est_psnr, est_ssim = obj_est

                fig, ax = plt.subplots(1, 5, figsize=(48, 12))
                fontsize = 18
                
                ax[0].imshow(y_batch[0].detach().cpu().squeeze(), cmap='gray')
                ax[0].axis('off')
                ax[0].set_title('Real Measurement', fontsize=fontsize)
                
                ax[1].imshow(y[0].detach().cpu().squeeze(), cmap='gray')
                ax[1].axis('off')
                ax[1].set_title('Sim Measurement', fontsize=fontsize)
                
                ax[2].imshow(I_est.detach().cpu().squeeze(), cmap='gray')
                ax[2].axis('off')
                ax[2].set_title('Object Estimate', fontsize=fontsize)
                
                ax[3].imshow(sim_angle, cmap='rainbow', vmin=-np.pi, vmax=np.pi)
                ax[3].axis('off')
                ax[3].set_title(f'Aberration Estimate', fontsize=fontsize)
                
                ax[4].imshow(_kernel[0].detach().cpu().squeeze(), cmap='viridis')
                ax[4].axis('off')
                ax[4].set_title('Sim post-SLM PSF', fontsize=fontsize)
                
                plt.savefig(f'{vis_dir}/e_{epoch}_it_{it}.png')
                plt.clf()
                sio.savemat(f'{vis_dir}/Sim_Phase.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})
                
                logger.info(f"Test of Objects Estimation at ({epoch} / {args.num_epochs}) |\t DISTS: {obj_est[0]:.4f}   LPIPS: {obj_est[1]:.4f}   PSNR: {obj_est[2]:.3f}   SSIM: {obj_est[3]:.4f}")
                logger.info("="*50)

            total_it += 1

        im_sche.step()
        ph_sche.step()

    t1 = time.time()
    print(f'Training takes {t1 - t0} seconds.')
    logger.info(f'Training takes {t1 - t0} seconds.')
    
    
    # ====================== Export Final Results ======================
    cur_t = torch.FloatTensor([-0.5]).to(DEVICE)
    
    I_est, sim_g, sim_phs = net.get_estimates(cur_t)
    sim_conj = torch.conj(sim_g[0].detach().cpu().squeeze())
    
    I_est_ = torch.clamp(I_est, 0, 1).squeeze()
    I_est  = np.uint16(I_est_.squeeze().detach().cpu().numpy() * 65535)
    
    est_g = sim_g[0].detach().cpu().squeeze().numpy()
    phs_err = np.uint8(ang_to_unit(CTF * np.angle(est_g)) * 255)
    
    abrr = CTF * (sim_phs[0].detach().cpu().squeeze())
    abrr_raw = abrr
    abrr = (abrr - abrr.min()) / (abrr.max() - abrr.min())
    abrr = np.uint8(abrr * 255)
    
    # plt.imsave(f'{vis_dir}/final/uncorrected_measurement.png', measure, cmap='gray')
    plt.imsave(f'{vis_dir}/final/GT_system_correction.png', target.numpy(), cmap='gray')
    plt.imsave(f'{vis_dir}/final/final_object_estimate.png', I_est, cmap='gray')
    tifffile.imwrite(f'{vis_dir}/final/final_object_estimate.tif', I_est)
    # plt.imsave(f'{vis_dir}/final/final_optical_correction.png', corr_res, cmap='gray')
    
    plt.imsave(f'{vis_dir}/final/final_aberrations_angle.png', phs_err, cmap='rainbow')
    plt.imsave(f'{vis_dir}/final/final_aberrations.png', abrr, cmap='rainbow')
    tifffile.imwrite(f'{vis_dir}/final/final_aberrations_raw.tif', abrr_raw.numpy())
    
    final_obj_est = evaluate_iqa_metric(I_est_, target, iqa_lists)
    

    print(f"Final Test of Objects Estimation  |\t DISTS: {final_obj_est[0]:.4f}   LPIPS: {final_obj_est[1]:.4f}   PSNR: {final_obj_est[2]:.3f}   SSIM: {final_obj_est[3]:.4f}")
    logger.info(f"Final Test of Objects Estimation  |\t DISTS: {final_obj_est[0]:.4f}   LPIPS: {final_obj_est[1]:.4f}   PSNR: {final_obj_est[2]:.3f}   SSIM: {final_obj_est[3]:.4f}")
    
    fig, ax = plt.subplots(1, 3, figsize=(25, 10))
    fontsize = 20
                
    ax[0].imshow(target.cpu().numpy()[0][0], cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Ground Truth', fontsize=fontsize)
                
    ax[1].imshow(I_est, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title(f'Object Estimate', fontsize=fontsize)
                
    ax[2].imshow(phs_err, cmap='rainbow')
    ax[2].axis('off')
    ax[2].set_title(f'Aberration Estimate', fontsize=fontsize)
    
    plt.savefig(f'{vis_dir}/final/results.png')
    plt.clf()
    plt.show()
    
    print("Training Concludes.")
    logger.info("Training Concludes.")
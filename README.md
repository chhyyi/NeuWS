# Neural Wavefront Shaping
I've implemented tif stack and added some output files.  ​
See [original repo's readme](https://github.com/Intelligent-Sensing/NeuWS/) for details.  

***Usage:***
```bash
python recon_exp_data_tif.py --num_t 100 --static_phase --scene_name v2_hw_stack_300um_rescale100-1k --phs_layers 8 --num_epochs 1000 --save_per_frame \
    --ims_pth neuws_exp_data/hw_stack/300umtotalim.tiff \
    -mod_pth neuws_exp_data/hw_stack/lowaberrationmap.tif \
    --rescale_min 100 \
    --rescale_max 1000 \
    --circle_mask 1.0 \
    --phase_rot 90
```  
about arguments of first lines, see original readme.  
* rescale : rescale measurement linearly in range [recale_min, rescale_max]  
* ims_pth : measurements images as tif stack. It will skip first and second stack cuz accrording to the data given to me by HW Jin.  
* mod_pth : mod, tif stack, in rad, wrapped in range $[-\pi, \pi]$
* circular_mask: multiply mask to pass only 'inside the circle' values.... working as CTF_size. if 1.0, radius of circle is equal to image size.  
* rotate_phase : rotate phase for some degrees... I tried only multiples of 90deg. I added this in worry of mis-rotated phase map but I don't think so...
* data_root configure was replaced by two(ims, mod) paths


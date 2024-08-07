# python main_syn.py --root_dir ./brain_data --data_dir low_pass_filtered_GT.tif \
#                 --num_t 100 --phs_layer 8 --num_epochs 1000 \
#                 --coeff_std_factor 4 --batch_size 32

# python main_syn.py --root_dir ./brain_data --data_dir low_pass_filtered_GT.tif \
#                 --num_t 100 --phs_layer 8 --num_epochs 1000 \
#                 --coeff_std_factor 2 --batch_size 32 \
#                 --coherent_avg

CUDA_VISIBLE_DEVICES=0 python main_real.py --root_dir ./real_data --data_dir dataset \
                                           --num_t 100 --phs_layer 8 --num_epochs 1000 --batch_size 32 \
                                           --depth 100

# python main_syn.py --root_dir ./brain_data --data_dir GT.tif \
#                 --num_t 100 --phs_layer 8 --num_epochs 1000 \
#                 --coeff_std_factor 3 --batch_size 32 --rand_abrr

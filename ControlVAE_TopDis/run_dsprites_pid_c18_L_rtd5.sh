#! /bin/sh

python3 main.py --train True --dataset dsprites --seed 1 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model HL --batch_size 64 --z_dim 10 --max_iter 1e6 \
    --C_stop_iter 5e5 --step_val 0.15 --gpu 0 \
    --viz_name $1 --C_max 18 \
    --use_rtd --lp 1 --q_normalize --gamma_rtd 5 --sample_based --weightnorm_sampler --delay_iter 1e5

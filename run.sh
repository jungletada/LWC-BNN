python run_bnn.py \
    --model-type U-Net  \
    --posterior-method SVI \
    --batch_size 64 \
    --num_samples 50 \
    --num_epochs 10000 \
    --return_height True \

# python run_bnn.py \
#     --model-type U-Net  \
#     --posterior-method MCMC \
#     --num_samples 50 \
#     --return_height True \
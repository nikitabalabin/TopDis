# DAVA: Disentangling Adversarial Variational Autoencoder
## Description
This code supports the ICLR 2023 Submission DAVA: Disentangling Adversarial Variational Autoencoder

## Getting Started
### Dependencies
We provide two environment files, `dava.yml` and `pipe.yml`. Due to versioning conflicts with disentanglement-lib,
we suggest to use separate environments  for training DAVA and evaluating the PIPE metric.
To our knowledge, for these two environments, no issues should arise when evaluating a model in a different environment than it was trained.
We provide dsprites as an example dataset in the correct format.

## DAVA
To reproduce DAVA:
 ``python dava/train_dava.py --dataset_path=datasets/dsprites.npz --z_dim=10 --batch_size=128 --num_steps=150000 --num_channels=1
 --store_path=~/dava/ --disc_weight=0.3 --c_max=8 --iteration_threshold=12800000 --disc_instance_norm 
 --gamma=500 --random_seed=0 --max_grad_norm=1 --learning_rate=0.0001 --disc_max_grad_norm=1
  --disc_learning_rate=0.0001 --use_mixed_batches --model_scale_factor=1 --start_kl_step=0
   --disc_confidence_value=0.6 --disc_decoder_weight=0.001``
   
To reproduce any of the other models, for example FactorVAE:
``python dava/train_factor_vae.py --batch_size=128 --num_channels=1 --learning_rate=0.0001 --dataset_path=datasets/dsprites.npz
--store_path=~/factor_vae/ --gamma=5 --num_steps=150000 --random_seed=0 --max_grad_norm=1``

## PIPE Metric
Reproducing the results of the PIPE metric is a bit more intricate. The code for computing the PIPE metric is provided in `pipe_metric.py`
Models were trained using [disentanblement-lib](https://github.com/google-research/disentanglement_lib). 
We used `dlib_reproduce` to train Beta-TCVAEs and FactorVAEs with 6 hyperparameters and 5 random seeds each.
We did this for each dataset considered. We then used disentanglement-lib to compute all supervised as well as the UDR metric.
We extended disentanglement-lib so that with slight changes, it can be used to evalaute PIPE as well.
We provide the necessary files here. `pipe_metric.py` needs to be placed in `disentanglement_lib.evaluation.metrics`.
`evaluate.py` replaces `disentanglement_lib.evaluation.evaluate.py`.
`compute_metrics_vae.py` gives an overview on how to start the metric computation. 
This works for both models reproduced with disentanglement-lib and models trained in the DAVA environment.

We further used disentanglement-lib to reproduce the results of the abstract reasoning study for BetaTCVAE and FactorVAE.
Models there could be evaluated using the same procedure as explained before.


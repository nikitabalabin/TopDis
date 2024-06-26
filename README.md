# Disentanglement Learning via Topology
This is an implementation of the algorithms from the paper [https://arxiv.org/pdf/2201.00058](https://arxiv.org/pdf/2308.12696)

## Description

The code contains implementation of the proposed approach TopDis.
For base models Beta-VAE, Beta-VAE, ControlVAE, FactorVAE, DAVA we use official implementations and add TopDis terms to each of the model.
In general, we added TopDis regularization to solver.py and code (rtd.py, rtd_regularizer.py) to compute TopDis loss.

### Installation
1. Install ripserplusplus:
```pip install git+https://github.com/simonzhang00/ripser-plusplus.git```
2. Install RTD:
```pip install git+https://github.com/IlyaTrofimov/RTD.git```
3. Requires patched ripserplusplus for RTD optimization:
```pip install git+https://github.com/ArGintum/RipserZeros.git@rtd-version```
4. Install Giotto
```pip install giotto-ph```

## Usage Example

0. Choose a method and change directory to one of the methods, e.g FactorVAE_TopDis
1. Download dataset from [here](https://drive.google.com/file/d/1Tp8eTdHxgUMtsZv5uAoYAbJR1BOa_OQm) and put it into `data` directory
2. File run_mpi3d_complex.sh contains the command with the set of hyperparameters that we used in our experiments.
These files can be executed using the command: 
```sh run_mpi3d_complex.sh <experiment_name>```
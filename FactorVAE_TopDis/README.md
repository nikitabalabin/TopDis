# Disentanglement Learning via Topology

## Description

The code contains implementation of the proposed approach FactorVAE + TopDis.
For the FactorVAE model, we used the code from https://github.com/1Konny/FactorVAE 
(files dataset.py, main.py, model.py, ops.py, solver.py, utils.py) and made several changes.
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

## Usage

The preparatory work and general usage correspond to the description from https://github.com/1Konny/FactorVAE.

1. Download dataset from [here](https://drive.google.com/file/d/1Tp8eTdHxgUMtsZv5uAoYAbJR1BOa_OQm) and put it into `data` directory
2. File run_mpi3d_complex.sh contains the command with the set of hyperparameters that we used in our experiments.
These files can be executed using the command: 
```sh run_mpi3d_complex.sh <experiment_name>```
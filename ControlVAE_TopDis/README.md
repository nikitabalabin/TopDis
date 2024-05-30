## Description

The code contains implementation of the proposed approach ControlVAE + TopDis.
For the ControlVAE model, we used the code from https://github.com/shj1987/ControlVAE-ICML2020

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

The preparatory work and general usage correspond to the description from https://github.com/shj1987/ControlVAE-ICML2020.

1. Download dataset 
```bash prepare_data.sh dsprites ```
2. File run_dsprites_pid_c18_L_rtd5.sh contains the command with the set of hyperparameters that we used in our experiments.
These files can be executed using the command: 
```sh run_dsprites_pid_c18_L_rtd5.sh <experiment_name>```
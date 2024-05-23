# Natural language processing course 2023/24: `Qualitative Research on Discussions`

Structure:
- data folder contains dataset used by code
- images folder contains images used in report
- TODO

## Installing packages
Use pip to install requirements.txt file in your virtual environment. For .ipynb file u can create one with VSC extension for Jupyter Notebooks and then run `!pip install -r requirements.txt` in the first cell to install all the packages. If you want to create a virtual env manually you can do so with python:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Note: 
### Recommended package installation is through terminal, not Jupyter Notebook. Use requirements.txt in root folder to install packages required for all modules. If you only want to run specific module, eg. baseline bert, use requirements.txt in folder @/baseline_bert to install required packages. 



Alternative is to use Conda module on SLURM:
```
module load Anaconda3/2023.07-2
conda create --name pytorch_env python=3.10
conda init bash
conda activate pytorch_env
```
This can be later used while running `srun` to load env:

```
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate pytorch_env
```



## General information
- Notebook corpus_analysis.ipynb is used to analyse the dataset
- Dataset is available in `./data` folder
- Load the dataset with pickle:
```
discussion_data = pd.read_pickle(corp_path + '/discussion_data.pkl')
```
- The baseline model can be found in the `baseline_bert` folder in the notebook `Baseline BERT.ipynb`. This folder also includes the pickle files containing the train, test and validation datasets.

## Running slurm job
- Use `test_run.sh` as a template for any scripts
- Run job with `sbatch run-slurm.sh`


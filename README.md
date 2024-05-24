# Natural language processing course 2023/24: `Qualitative Research on Discussions`

Structure:
- `data` folder contains dataset used by code
- `images` folder contains images used in report
- `report` folder contains report
- `traditional_AI_approaches` folder contiains an implementation of Traditional AI approaches and its evaluation
- `baseline_bert` folder contiains an implementation of BERT model and its evaluation
- `prompt_tuning` folder contiains an implementation of text categorization by prompting and its evaluation
- `corpus_anaylsis.ipynb` file is a Jupyter notebook file for corpus analysis
- `start_jupyter.job` is a script to start Jupyter notebook on SLURM for all jupyter notebooks in root directory

## Installing packages
Use pip to install requirements.txt file in your virtual environment. For .ipynb file u can create one with VSC extension for Jupyter Notebooks and then run `!pip install -r requirements.txt` in the first cell to install all the packages. If you want to create a virtual env manually you can do so with python:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and set a configuration of Jupyter Notebook
```
pip install jupyter
jupyter notebook --generate-config
jupyter notebook password
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

### Corpus analysis
- Notebook corpus_analysis.ipynb is used to analyse the dataset
- Dataset is available in `./data` folder
- Load the dataset with pickle:
```
discussion_data = pd.read_pickle(corp_path + '/discussion_data.pkl')
```
### Baseline BERT
- The baseline model can be found in the `baseline_bert` folder in the notebook `Baseline BERT.ipynb` and in notebook `Baseline BERT - longer training.ipyb`. This folder also includes the pickle files containing the train, test and validation datasets.

### Distil BERT
- The model DistilBERT can be found in the `baseline_bert` folder in the notebook `DistilBERT.ipynb`. This folder also includes the pickle files containing the train, test and validation datasets.

### Traditional AI approaches
- In the folder `traditional_AI_approaches` in the notebook `Traditional AI approaches to text classification.ipynb` you can run the traditional AI models. In this folder you can also run the error analysis notebook to get a more thorough look at the model's results.

### Prompting
- Text categorization with prompt engineering can be found in the `prompt_tuning` folder the notebook `prompt_tuning.ipynb`. This folder also includes the pickle files containing the train, test and validation datasets.
- **KEY NOTE**: You need to add file `.env` which includes your hugging face token `HF_TOKEN=<your token>`. You also have to apply for using a model [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) beforehand.

## Running slurm job
- Move to directory you want to execute Jupyter Notebook
- Run job with `sbatch start_jupyter.job`
- The script creates a log folder with files where you can find a link to a running notebook

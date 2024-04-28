import pandas as pd
import os
from dotenv import load_dotenv

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

load_dotenv()  # This loads the .env file into the environment

hf_token = os.getenv('HF_TOKEN')

discussion_data = pd.read_pickle(r'./data/discussion_data.pkl')

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_name = 'meta-llama/Llama-2-13b-chat-hf'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # loading in 4 bit 
    bnb_4bit_quant_type="nf4", # quantization type
    bnb_4bit_use_double_quant=True, # nested quantization 
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    token=hf_token
)
model.config.use_cache = False
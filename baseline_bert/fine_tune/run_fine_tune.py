import torch
import pandas as pd
import pickle
from datasets import Dataset
import torch
from datasets import DatasetDict
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import DataCollatorWithPadding
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np


BATCH_SIZE = 16
EPOCHS = 20
bert_path = '../'

train_df = pd.read_pickle(bert_path+"train_dataset")
test_df = pd.read_pickle(bert_path+"test_dataset")
val_df = pd.read_pickle(bert_path+"val_dataset")

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(test_df),
    'unsupervised': Dataset.from_pandas(val_df)
})

# define preprocess function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

tokenized_data = dataset.map(preprocess_function, batched=True, batch_size=100,  load_from_cache_file=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

print(device, n_gpu)
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")


model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=7)

model.to('cuda')

training_args = TrainingArguments(
    output_dir="./bert_runs",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1
)

optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
# Use LR Plateau to reduce learning rate when the model stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

training_args.optimizer = optimizer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)

torch.cuda.empty_cache()

trainer.train()

print("Training finished\n\n\n\n")

print("Test scores: ", trainer.evaluate())


print("Validation scores: ", trainer.predict(tokenized_data['unsupervised']))
import ast
from collections import Counter
import io
import numpy as np
import pandas as pd
import requests
import torch
from transformers import RobertaConfig, RobertaTokenizer, TrainingArguments
from preprocessing import preprocess
import Roberta_PCL
import PCLDataset
import Trainer_PCL
import evaluation

GPU = True # Choose whether to use GPU
if GPU:
    DEVICE = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")
    
# Load in train/test ids
train_ids_url =  # Make sure the url is the raw version of the file on GitHub
test_ids_url = # Make sure the url is the raw version of the file on GitHub

download_train_val = requests.get(train_ids_url).content
download_test = requests.get(test_ids_url).content

# Reading the downloaded content and turning it into a pandas dataframe
train_val_ids = pd.read_csv(io.StringIO(download_train_val.decode('utf-8')))
test_ids = pd.read_csv(io.StringIO(download_test.decode('utf-8')))

# Convert paragraph IDs to strings
train_val_ids.par_id = train_val_ids.par_id.astype(str)
test_ids.par_id = test_ids.par_id.astype(str)

# Convert labels to lists
train_val_ids.label = [ast.literal_eval(i) for i in train_val_ids.label]

# Convert One-hot Vectors to category information in Data Frames
categories = ['power_imbalance', 'compassion', 'presupposition', 'authority', 'shallow_solution', 'metaphor', 'poorer_merrier']

# Create columns for each category
for cat in categories:
  train_val_ids[cat] = 0

# Change one hot vectors to columns
counter = 0
for label in train_val_ids.label:
  for col, cat in enumerate(label):
    train_val_ids[categories[col]][counter] = cat
  counter += 1
train_val_ids.head(10)

# Load all data in
"""
		Paragraphs with original labels of 0 or 1 are considered to be negative examples of PCL and will have the label 0 = negative.
		Paragraphs with original labels of 2, 3 or 4 are considered to be positive examples of PCL and will have the label 1 = positive.
"""
dpm = DontPatronizeMe('.', '.')
dpm.load_task1()
data = dpm.train_task1_df


train_val_set = pd.merge(train_val_ids[['par_id','power_imbalance', 'compassion', 'presupposition', 'authority', 'shallow_solution', 'metaphor', 'poorer_merrier']], data, how='inner', on='par_id')
test_set = pd.merge(test_ids['par_id'], data, how='inner', on='par_id')
# Add column to include input length of the sentence
train_val_set['Input Length'] = train_val_set['text'].apply(len)

# Split the Training set into training and validation set in 80/20 split
train_set, val_set = np.split(train_val_set.sample(frac=1), [int(0.8 * len(train_val_set))])

# Only keep the text and label for training
train_data = train_set[['text', 'label']].rename(columns={"text": "text", "label": "orig_labels"})
val_data = val_set[['text', 'label']].rename(columns={"text": "text", "label": "orig_labels"})
test_data = test_set[['text', 'label']].rename(columns={"text": "text", "label": "orig_labels"})

def make_probas(row):
  # if label = 'Not PCL' / 0 then return probabilities 1, 0
  if row == 0:
    return np.array([1.0,0.0])
  # if label = 'PCL' / 1 then return probabilities 0, 1
  if row == 1:
    return np.array([0.0,1.0])

# Convert back
def get_orig_label(row):
  # if label = 'Not PCL' / 0 then return probabilities 1, 0
  if np.all(row == np.array([1.0,0.0])):
    return 0.0
  # if label = 'PCL' / 1 then return probabilities 0, 1
  if np.all(row == np.array([0.0,1.0])):
    return 1.0
    
train_data['labels'] = train_data.orig_labels.apply(make_probas)
val_data['labels'] = val_data.orig_labels.apply(make_probas)
test_data['labels'] = test_data.orig_labels.apply(make_probas)

# Apply preprocessing steps and augmentation to train dataset
train_data_prep_upsampled3 = preprocess(train_data, data_aug=True, class_value=1, scale=3, n_replaced=15)
num_pos = len(train_data_prep_upsampled3[train_data_prep_upsampled3.orig_labels==1])

# Downsample positive class
train_data_up_down3 = pd.concat([train_data_prep_upsampled3[train_data_prep_upsampled3.orig_labels==1],train_data_prep_upsampled3[train_data_prep_upsampled3.orig_labels==0][:num_pos]])

print(Counter(train_data_up_down3.orig_labels.values))
train_data_up_down3 = train_data_up_down3[['text', 'labels']]
val_data_preprocessed = preprocess(val_data)

# Function to instantiate model with different hyperparameters and then save model
def main(model_config, 
         dropout_rate, 
         train_dataset, 
         val_dataset,
         model_name='Roberta', 
         tokenizer=None,
         batch_size=32, 
         num_epochs=5,
         eval_strat='steps', 
         logging_steps=20,
         learning_rate=0.00001,
         warmup_ratio=0.0,
         label_smoothing_factor=0.0,
         lr_scheduler_type='linear'): 

    # Instantiate Model with desired model_config
    model = Roberta_PCL(model_config, dropout_rate)

    # Create PCL Dataset objects from input datasets and tokenizer
    if tokenizer == None:
      tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    # Make sure the data is shuffled for training
    model_train_dataset = train_dataset.sample(frac = 1)
    model_val_dataset = val_dataset.sample(frac = 1)

    # Convert Dataset into Datasets
    model_train_dataset = PCLDataset(tokenizer, model_train_dataset.to_dict(orient = 'list'))
    model_val_dataset = PCLDataset(tokenizer, model_val_dataset.to_dict(orient = 'list'))

    # Instantiate Model Training arguments    
    training_args = TrainingArguments(
        output_dir='./base_case/pcl',
        learning_rate = learning_rate,
        logging_steps= logging_steps,
        eval_steps=logging_steps,
        do_train=True,
        do_eval=True,
        evaluation_strategy=eval_strat,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs = num_epochs,
        warmup_ratio=warmup_ratio,
        label_smoothing_factor=label_smoothing_factor,
        lr_scheduler_type = lr_scheduler_type)

    # Instantiate Trainer with desired arguments
    trainer = Trainer_PCL(
        model=model,                         
        args=training_args,                
        train_dataset=model_train_dataset,
        eval_dataset=model_val_dataset,
        data_collator=model_train_dataset.collate_fn)

    # Train Model on Training dataset
    trainer.train()

    # Save Model
    trainer.save_model(f'./models/{model_name}/')
    return trainer
    
    
# Roberta Model with upsampling and downsampling (preprocessing), ratio 50:50, more samples
# Initial Hyperparameters
config = RobertaConfig()
config.vocab_size = 50265
model_name = 'model'
dropout_rate = 0.2
batch_size = 8
num_epochs = 3
lr = 1e-5

# Train model
trained = main(config, 
                dropout_rate,
                train_data_up_down3, # Use upsampled training data
                val_data_preprocessed, # Use validation data without preprocessing
                model_name=model_name,
                batch_size=batch_size, 
                num_epochs=num_epochs, 
                eval_strat='steps', 
                logging_steps=20, 
                learning_rate=lr)
    
# Plot loss curves
evaluation.plot_loss_curves(trained.state.log_history, 'Model')

# Evaluate model on whole validation set
model = Roberta_PCL.from_pretrained(f'./models/{model_name}/', dropout_rate)
evaluation.evaluate(model, val_data_preprocessed)



    
    


import ast
from collections import Counter
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix
import string
import random
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaModel,  RobertaTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, ElectraModel, ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification
from transformers.utils.dummy_pt_objects import RobertaPreTrainedModel, BertPreTrainedModel
from urllib import request


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

#Preprocessing 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet 

def remove_punc(row):
  punctuation = string.punctuation
  for p in punctuation:
    row = row.replace(p, '')
  return row

# Function to make string lowercase
def lowercase(row):
  row = row.lower()
  return row

# Function to remove numbers from a string
def remove_numbers(row):
  row = ''.join(word for word in row if not word.isdigit())
  return row  

# Function to remove stop words from a string
def remove_stop_words(row):
  stop_words_list = set(stopwords.words('english'))
  word_tokens = word_tokenize(row)
  row = " ".join([w for w in word_tokens if not w in stop_words_list])
  return row
 
# Function to get synonyms of a word
def get_synonyms(word):

    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)
    
 def synonym_replacement(words, n):
    
    words = words.split()
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n: # only replace up to n words
            break

    sentence = ' '.join(new_words)
    return sentence  
    
    # Function to augment positive class by adding sentences with synonyms 

def augment_class(train_dataset, class_value=1, scale=5, n_replaced=5):

  # Print class ratio before augmentation
  print("Number of positive and negative classes before agumentation:")
  print(train_dataset['orig_labels'].value_counts())

  # Get PCL samples (i.e. where label == 1)
  dataset_copy = train_dataset.copy()
  pcl_samples = dataset_copy[dataset_copy.orig_labels == class_value]
  
  # Augment samples 'scale' times
  all_dfs = [train_dataset]
  for iter in range(scale-1):
    pcl_samples_copy = pcl_samples.copy()
    pcl_samples_copy['text'] = pcl_samples_copy['text'].apply(synonym_replacement, args=(n_replaced,)) # 'n_replaced' = number of words to be replaced with synonyms
    pcl_samples_copy['orig_labels'] = class_value
    all_dfs.append(pcl_samples_copy)

  # Add to original training dataset and print new class ratios
  augmented_df = pd.concat(all_dfs)
  print("Number of positive and negative classes after augmentation:")
  print(augmented_df['orig_labels'].value_counts())
  augmented_df['labels'] = augmented_df['orig_labels'].apply(make_probas)

  return augmented_df
  
# Augment the positive class and downsample negative class (1:1 ratio) with preprocessing, more samples

# Apply preprocessing steps and augmentation to train dataset
train_data_prep_upsampled3 = preprocess(train_data, data_aug=True, class_value=1, scale=3, n_replaced=15)
num_pos = len(train_data_prep_upsampled3[train_data_prep_upsampled3.orig_labels==1])

# Test how many rows are identical i.e. whether synonym replacement rate is sufficient
test_augmentation(train_data_prep_upsampled3)

# Downsample positive class
train_data_up_down3 = pd.concat([train_data_prep_upsampled3[train_data_prep_upsampled3.orig_labels==1],train_data_prep_upsampled3[train_data_prep_upsampled3.orig_labels==0][:num_pos]])

print(Counter(train_data_up_down3.orig_labels.values))
train_data_up_down3 = train_data_up_down3[['text', 'labels']]

class PCLDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, input_set):

        self.tokenizer = tokenizer
        self.texts = input_set['text']
        self.labels =input_set['labels']
  
    def collate_fn(self, batch):

        texts = []
        labels = []

        for b in batch:
            texts.append(b['text'])
            labels.append(b['labels'])

        encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        encodings['labels'] =  torch.tensor(labels)
        
        return encodings
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
       
        item = {'text': self.texts[idx],
                'labels': self.labels[idx]}
        return item


# Roberta Class to Instantiate Model and define Forward Pass

class Roberta_PCL(RobertaModel):

    def __init__(self, config, dropout_rate):
        super().__init__(config)

        # RoBertaModel
        self.roberta = RobertaModel(config)
        
        # Linear Layer for Binary Classification output
        self.linear = torch.nn.Sequential(torch.nn.Dropout(dropout_rate),
                                                torch.nn.Linear(config.hidden_size, 2),
                                                torch.nn.Softmax(dim=1))  
        
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

 
        # Forward pass through Roberta Model
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Forward pass through Linear Layer for Binary Classification
        logits = self.linear(outputs[1])
        
        return logits
# Define Trainer Object for training model

class Trainer_PCL(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        # Remove Labels from input
        labels = inputs.pop('labels')

        # Forward pass input
        outputs = model(**inputs)

        # Use CrossEntropyLoss as error
        criterion = torch.nn.CrossEntropyLoss().to('cuda')

        loss = criterion(outputs.view(-1, 2), labels.view(-1, 2)) 

        return (loss, outputs) if return_outputs else loss

# Function to plot loss curves from a trained model history

def plot_loss_curves(history, model_name):

  epochs = []
  steps = []
  train_losses = []
  val_losses = [] 

  # Get log items from history dictionary
  for i, log in enumerate(history[:-1]): # get rid of last log entry with no eval loss
    if i % 2 == 0:
      epochs.append(log['epoch'])
      steps.append(log['step'])
      train_losses.append(log['loss']) 
    elif i % 2 != 0:
      val_losses.append(log['eval_loss']) 
    
  # Plot figure
  plt.figure(figsize=(10,6))
  plt.plot(steps, train_losses, label='Training loss')
  plt.plot(steps, val_losses, label='Validation loss')
  plt.title(f'Loss Curves for {model_name}')
  plt.ylabel('Loss (cross-entropy)')
  plt.xlabel('Steps')
  plt.legend()
  
def predict_pcl(input, tokenizer, model): 
  model.eval()
  encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=128)
  output = model(**encodings)
  preds = torch.max(output, 1)
  return {'prediction':preds[1], 'confidence':preds[0]}
  
# Function to calculate F1 score during training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    return {"PCL (Class 1) F1": f1}
  
# Function to generate evaluation metrics of a model

def evaluate(model, eval_set, tokenizer=None, evaluate_wrong_predictions=True, evaluation_set="VALIDATION", binary_labels=False, electra_2=False):

  if tokenizer==None:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')


  val_set_copy = eval_set.copy()

  if not binary_labels:
      val_set_copy['labels'] = val_set_copy['labels'].apply(get_orig_label)
  
  val_dataloader = DataLoader(PCLDataset(tokenizer, val_set_copy.to_dict(orient = 'list')))
    

  preds = []
  tot_labels = []
  wrong_predictions = []

  with torch.no_grad():
    for data in tqdm(val_dataloader): 

      labels = {}
      labels['labels'] = data['labels']

      text = data['text']

      if not binary_labels:
          pred = predict_pcl(text, tokenizer, model)
      
      elif binary_labels:
          if electra_2:
            pred = predict_pcl_electra(text, tokenizer, model)
          else:
            pred = predict_pcl_2(text, tokenizer, model)

      if pred['prediction'] != labels['labels']:
        wrong_predictions.append(text)
      
      preds.append(pred['prediction'].tolist())
      tot_labels.append(labels['labels'].tolist())
  

  # with the saved predictions and labels we can compute accuracy, precision, recall and f1-score
  report = classification_report(tot_labels, preds, target_names=["Not PCL","PCL"])
  cm = confusion_matrix(tot_labels, preds)

  print('\n')
  print(f"EVALUATION METRICS FROM {evaluation_set} DATA")
  print("----------------------------------------------------------")
  print(report)
  print('\n')
  print(f"CONFUSION MATRIX FROM {evaluation_set} DATA")
  print("----------------------------------------------------------")
  print(cm)
  print('\n')

  if evaluate_wrong_predictions:
      wrong_prediction_indices = []

      for wrong_pred in wrong_predictions:
        index = eval_set.index[eval_set['text'] == wrong_pred[0]]
        wrong_prediction_indices.append(index[0])

      wrong_preds_df = val_set[val_set.index.isin(wrong_prediction_indices)]
      print(f"SAMPLE OF 5 WRONG PREDICTIONS FROM {evaluation_set} DATA:")
      print("----------------------------------------------------------")
      for p in wrong_preds_df.text.values[:5]:
        print(p)
  return  
# Function to generate evaluation metrics of a model

def test_predict(model, eval_set, tokenizer):

  val_set_copy = eval_set.copy()  
  val_dataloader = DataLoader(PCLDataset(tokenizer, val_set_copy.to_dict(orient = 'list')))
    
  preds = []

  with torch.no_grad():
    for data in tqdm(val_dataloader): 

      labels = {}
      text = data['text']
      pred = predict_pcl_2(text, tokenizer, model)
      
      preds.append(pred['prediction'].tolist())
    
  return preds
  
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
plot_loss_curves(trained.state.log_history, 'Model')

# Evaluate model on whole validation set
model = Roberta_PCL.from_pretrained(f'./models/{model_name}/', dropout_rate)
evaluate(model, val_data_preprocessed)



    
    


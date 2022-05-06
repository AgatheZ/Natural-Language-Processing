import matplotlib.pyplot as plt
import torch 
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import RobertaTokenizer
import PCLDataset
from torch.utils.data import DataLoader
import tqdm

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
  
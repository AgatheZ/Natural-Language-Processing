import string
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from main import make_probas

#Preprocessing 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

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

def preprocess(dataset, remove_punct=True, number_removal=True, case=True, stop_words=True, data_aug=False, class_value=1, scale=5, n_replaced=5):

    if remove_punct:
      dataset['text'] = dataset['text'].apply(remove_punc)
    
    if number_removal:
      dataset['text'] = dataset['text'].apply(remove_numbers)
    
    if case:
      dataset['text'] = dataset['text'].apply(lowercase)

    if stop_words:
      dataset['text'] = dataset['text'].apply(remove_stop_words)

    if data_aug:
      dataset = augment_class(dataset, class_value=class_value, scale=scale, n_replaced=n_replaced)
        
    return dataset
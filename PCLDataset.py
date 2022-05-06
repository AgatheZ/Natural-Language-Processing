import torch 

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

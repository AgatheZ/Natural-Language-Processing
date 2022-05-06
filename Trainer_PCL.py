import torch
from transformers import Trainer

#Trainer Object for training model
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
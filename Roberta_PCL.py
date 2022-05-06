
from transformers import RobertaModel
import torch

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

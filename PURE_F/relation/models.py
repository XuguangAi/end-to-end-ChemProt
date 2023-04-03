import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import AlbertModel, AlbertPreTrainedModel

from allennlp.modules import FeedForward
from allennlp.nn.util import batched_index_select
import torch.nn.functional as F

BertLayerNorm = torch.nn.LayerNorm
class BertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 6)
        self.classifier = nn.Linear(config.hidden_size * 6, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_start_idx=None, sub_end_idx=None, obj_start_idx=None, obj_end_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]
        
        sub_start_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_start_idx)])
        sub_end_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_end_idx)])
        obj_start_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_start_idx)])
        obj_end_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_end_idx)])
        
        cls_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_start_idx - sub_start_idx)])
        
        middle_output = torch.tensor([])
        middle_output = middle_output.to(device='cuda')

        for a, i, j, k, l in zip(sequence_output, sub_start_idx, sub_end_idx, obj_start_idx, obj_end_idx):
            if j + 2 <= k: # Subject is before object and there is at least one token between them
                temp_tensor = a[i] - a[i]
                for t in range(k-j-1):
                    temp_tensor += a[j+1+t]
                temp_tensor = temp_tensor / (k-j-1)   
            elif l + 2 <= i: # Object is before subject and there is at least one token between them
                temp_tensor = a[i] - a[i]
                for t in range(i-l-1):
                    temp_tensor += a[l+1+t]
                temp_tensor = temp_tensor / (i-l-1)
            else:
                temp_tensor = a[i] - a[i] # The middle embedding is zero when there is no token between subject and object
            middle_output = torch.cat((middle_output, temp_tensor.unsqueeze(0)))
       
        rep = torch.cat((cls_output, sub_start_output, sub_end_output, middle_output, obj_start_output, obj_end_output), dim=1)
        # Relation representation F: [cls_output: sub_start_output: sub_end_output: middle_output: obj_start_output: obj_end_output]

        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class AlbertForRelation(AlbertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(AlbertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_start_idx=None, sub_end_idx=None, obj_start_idx=None, obj_end_idx=None):
        outputs = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False)
        sequence_output = outputs[0]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_start_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_start_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

from typing import Dict, Optional, List, Any

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy

############################################# 

#7/27- homework 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#input LSTM 

#######################################################


@Model.register("kevin_mod")
class kevin_mod(Model): 

    def __init__(self, hidden_dim:int, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 num_layers=1, bidirectional=FALSE, dropout_rate=0.0, num_classification=3,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)


        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1 
        self.hidden_dim = hidden_dim
        
        self.LSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, 
        bidirectional=bidirectional, dropout=dropout_rate) 

        self.fc = nn.Linear(hidden_dim * self.num_directions * self.num_layers * 2, hidden_dim*2)

        self.fc2= nn.Linear(hidden_dim*2 , hidden_dim*2)

        self.fc3= nn.Linear(hidden_dim*2, num_classification)
   



###################################################################################
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()


        _, (v1,_) = self.LSTM(embedded_premise) #these are independent 
        _, (v2,_) = self.LSTM(embedded_hypothesis)
    
#         v1 = torch.squeeze(v1, dim=0)
#         v2 = torch.squeeze(v2, dim=0)
        v1 = v1.view((-1, self.hidden_dim * self.num_directions * self.num_layers))
        v2 = v2.view((-1, self.hidden_dim * self.num_directions * self.num_layers))
        
        v1_cat_v2 = torch.cat((v1, v2), dim=1) # v1_cat_v2: (batch_size x (hidden_dim * 2))
        h = self.fc(v1_cat_v2)
        h = F.tanh(h)
        h = self.fc2(h)
        h = F.tanh(h)
        h = F.tanh(h)
        label_logits = self.fc3(h) 
        

        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)



        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
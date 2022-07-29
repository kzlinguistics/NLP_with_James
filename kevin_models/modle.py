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
    """
    This ``Model`` implements the ESIM sequence model described in `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
    by Chen et al., 2017.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    encoder : ``Seq2SeqEncoder``
        Used to encode the premise and hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between encoded
        words in the premise and words in the hypothesis.
    projection_feedforward : ``FeedForward``
        The feedforward network used to project down the encoded and enhanced premise and hypothesis.
    inference_encoder : ``Seq2SeqEncoder``
        Used to encode the projected premise and hypothesis for prediction.
    output_feedforward : ``FeedForward``
        Used to prepare the concatenated premise and hypothesis for prediction.
    output_logit : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)
    
    #this is to build the LSTM and the feed forward 
    #this is where I put in my LSTM layer and feed forward layer
    #figure 3 in SNLI 

#########################################################################

#7/27 added 

class LSTM_Module(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=True, dropout_rate=0.0):
        super(LSTM_Module, self).__init__()
        self.num_layers = num_layers
        self.num_directions = bidirectional + 1
        self.hidden_dim = hidden_dim
        
        self.LSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, 
        bidirectional=bidirectional, dropout=dropout_rate)

        self.fc = nn.Linear(hidden_dim * self.num_directions * self.num_layers * 2, hidden_dim)
        
    def forward(self, a, b):
        _, (_, v1) = self.LSTM(a)
        _, (_, v2) = self.LSTM(b)
        
#         v1 = torch.squeeze(v1, dim=0)
#         v2 = torch.squeeze(v2, dim=0)
        v1 = v1.permute(1, 0, 2).contiguous().view((-1, self.hidden_dim * self.num_directions * self.num_layers))
        v2 = v2.permute(1, 0, 2).contiguous().view((-1, self.hidden_dim * self.num_directions * self.num_layers))
        
        v1_cat_v2 = torch.cat((v1, v2), dim=1) # v1_cat_v2: (batch_size x (hidden_dim * 2))
        h = self.fc(v1_cat_v2)
        h = F.relu(h)
        
        return h



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

        label_logits = None #this is where I put in three calsses 
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
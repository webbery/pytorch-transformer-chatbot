import torch
from torch import nn

from data_utils.utils import Config
from data_utils.vocab_tokenizer import Vocabulary
from model.embedding.embeddings import Embeddings
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer, BertModel, AdamW, BertPreTrainedModel,get_linear_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BertDiscriminator(BertPreTrainedModel):
    '''
    使用bert来做分类器
    '''
    def __init__(self, config):
        super(BertDiscriminator, self).__init__(config)
        self.bert = BertModel(config)
        self.similarity = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            labels=None):
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = outputs[1] 
        cls_output = self.similarity(cls_output) # batch, 2
        # print(cls_output.shape)
        cls_output = torch.sigmoid(cls_output)
        criterion = nn.NLLLoss()
        # print(cls_output.shape,labels.shape)
        # print(labels)
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, torch.squeeze(labels))
        return loss, cls_output

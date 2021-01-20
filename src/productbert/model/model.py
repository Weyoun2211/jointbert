import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import BertModel, BertConfig, DistilBertModel, RobertaModel
from transformers.modeling_bert import BertOnlyMLMHead
from torch.cuda.amp import autocast



class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DistilBertModelLogit(BaseModel):

    def __init__(self, num_classes=1, freeze_bert=False):
        super().__init__()

        self.bert_layer = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        # Classification layer
        self.cls_layer = nn.Linear(768, num_classes)

        self._init_weights_bert(self.pre_classifier)
        self._init_weights_bert(self.cls_layer)


    @autocast()
    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        distilbert_output = self.bert_layer(seq, attention_mask=attn_masks)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.cls_layer(pooled_output)  # (bs, dim)

        return logits


class BertModelLogit(BaseModel):
    def __init__(self, num_classes=1, freeze_bert=False):
        super().__init__()

        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, num_classes)
        self._init_weights_bert(self.cls_layer)

    @autocast()
    def forward(self, seq, token_ids, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        # Feeding the input to BERT model to obtain contextualized representations
        _, pooler_output = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=token_ids)

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(pooler_output)

        return logits

class JointBertModelLogit(BaseModel):
    def __init__(self, num_classes_multi, num_classes=1, freeze_bert=False):
        super().__init__()

        self.num_classes_multi = num_classes_multi
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, num_classes)
        self.multi1_cls_layer = nn.Linear(768, num_classes_multi)
        self.multi2_cls_layer = nn.Linear(768, num_classes_multi)
        self.multi_softmax = nn.LogSoftmax(1)
        self._init_weights_bert(self.cls_layer)
        self._init_weights_bert(self.multi1_cls_layer)
        self._init_weights_bert(self.multi2_cls_layer)

    @autocast()
    def forward(self, seq, token_ids, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        # Feeding the input to BERT model to obtain contextualized representations
        _, pooler_output = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=token_ids)

        # Feeding cls_rep to the classifier layer
        logits_binary = self.cls_layer(pooler_output)
        logits_multi1 = self.multi1_cls_layer(pooler_output)
        logits_multi2 = self.multi2_cls_layer(pooler_output)

        multi1 = self.multi_softmax(logits_multi1)
        multi2 = self.multi_softmax(logits_multi2)

        return logits_binary, multi1, multi2

class RobertaModelLogit(BaseModel):
    def __init__(self, num_classes=1, freeze_bert=False):
        super().__init__()

        self.bert_layer = RobertaModel.from_pretrained('roberta-base')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, num_classes)
        self._init_weights_bert(self.cls_layer)

    @autocast()
    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        # Feeding the input to BERT model to obtain contextualized representations
        _, pooler_output = self.bert_layer(seq, attention_mask=attn_masks)

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(pooler_output)

        return logits
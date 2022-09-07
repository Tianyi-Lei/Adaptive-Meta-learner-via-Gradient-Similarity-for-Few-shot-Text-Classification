from torch import nn

from transformers import BertForSequenceClassification, BertConfig, BertForMaskedLM



class MaskedLMTransformerClassification(nn.Module):
    def __init__(self, args, device):
        super(MaskedLMTransformerClassification, self).__init__()

        self.num_labels = args.way
        self.bert_pretrain_path = args.bert_pretrain_path
        self.config = BertConfig.from_pretrained(self.bert_pretrain_path, num_labels=self.num_labels)
        self.cm = BertForSequenceClassification.from_pretrained(self.bert_pretrain_path, num_labels=self.num_labels)  # 主模型
        self.lm = BertForMaskedLM.from_pretrained(self.bert_pretrain_path)  # 辅助模型
        self.lm.bert.encoder = self.cm.bert.encoder
        # self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.cm.to(self.device)
        self.lm.to(self.device)
        self.dropout = nn.Dropout()
        self.fewrel = args.fewrel

    def forward(self, input_ids, attention_mask, classification_label_id, lm_label_ids, pos1, pos2):
        if self.fewrel:
            out_cm = self.cm(input_ids=input_ids, attention_mask=attention_mask, labels=classification_label_id)
            out_lm = self.lm(input_ids=input_ids, labels=lm_label_ids)
            cm_loss = out_cm[0]
            lm_loss = out_lm[0]
            return cm_loss, lm_loss, out_cm[1]
        else:
            out_cm = self.cm(input_ids=input_ids, attention_mask=attention_mask, labels=classification_label_id, output_hidden_states=True)
            out_lm = self.lm(input_ids=input_ids, labels=lm_label_ids)
            # hidden_state = out_cm.hidden_states
            # pooled_output = self.dropout(hidden_state)
            cm_loss = out_cm[0]
            lm_loss = out_lm[0]
            return cm_loss, lm_loss, out_cm[1]
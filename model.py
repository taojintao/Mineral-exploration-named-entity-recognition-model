import torch
from transformers.modeling_bert import *
#from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import torch.nn as nn


class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_heads = 8
        self.alpha = 0.9

        self.bert = BertModel(config)
        self.multiheadAttn= nn.MultiheadAttention(config.hidden_size,self.num_heads,dropout=0.0,batch_first=True)
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_embedding_size,
            batch_first=True,
            num_layers=2,
            dropout=config.lstm_dropout_prob,
            bidirectional=True
        )
        #ensure that the max_seq_len remains unchanged
        self.conv1ds=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=config.hidden_size,out_channels=config.cnn_embedding_size,kernel_size=h,
                          stride=1,padding=(h-1)//2,padding_mode='replicate'),
                nn.ReLU()
            )
            for h in config.window_sizes
        ]) # [3,5,7]


        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        input_dim=config.lstm_embedding_size*2+len(config.window_sizes)*config.cnn_embedding_size+config.hidden_size

        self.start = nn.Linear(input_dim, 1)
        self.end = nn.Linear(input_dim, 1)
        self.classifier = nn.Linear(input_dim, config.num_labels)
        self.loss_bd = nn.BCEWithLogitsLoss(reduction='none')
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, label_data=None,
                position_ids=None, inputs_embeds=None, head_mask=None,label_embedding=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        # Remove [CLS] and [SEP] labels, obtaining pre_labels aligned with the actual labels.
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # Pad the dimensions of the pred_label in the sequence_output to the maximum length
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        #origin_sequence_output = padded_sequence_output
        #ATTENTION
        # query (batch_size,L,Embed_dim_Q)
        # key (batch_size,S,Embed_dim_K)
        # value (batch_size,S,Embed_dim_V)
        multiheadAttn_output=self.multiheadAttn(padded_sequence_output, label_embedding, label_embedding)
        # (batch_size,L,Embed_dim_V)
        attn_output = multiheadAttn_output[0]

        #LSTM
        lstm_output, _ = self.bilstm(padded_sequence_output)

        #CNN
        padded_sequence_output=padded_sequence_output.permute(0,2,1)
        outputs=[conv1(padded_sequence_output) for conv1 in self.conv1ds]
        cnn_output=torch.cat(outputs,dim=1)
        cnn_output=cnn_output.permute(0,2,1)


        #OUTPUT_CAT
        output_cat=torch.cat([cnn_output,lstm_output,attn_output],dim=2)
        output_cat=self.dropout(output_cat)

        logits_start = torch.relu(self.start(output_cat))
        logits_end = torch.relu(self.end(output_cat))

        output_cat_new=torch.sigmoid(logits_start)*output_cat +torch.sigmoid(logits_end)*output_cat

        # Obtain the discriminative values
        logits = self.classifier(output_cat_new)
        outputs = (logits,)
        if label_data is not None:
            labels, starts, ends = label_data
            loss_mask = labels.gt(-1)
            loss_boundary = self.loss_bd(logits_start,starts.float().unsqueeze(2)) + self.loss_bd(logits_end,ends.float().unsqueeze(2))
            loss_boundary = torch.sum(loss_boundary * loss_mask.float().unsqueeze(2)) / torch.sum(loss_mask.float())

            loss_crf = self.crf(logits, labels, loss_mask) * (-1)

            loss = self.alpha*loss_crf + (1-self.alpha)*loss_boundary

            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs



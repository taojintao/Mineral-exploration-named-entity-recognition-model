import torch
from transformers import BertTokenizer,BertModel
import config


tokenizer = BertTokenizer.from_pretrained(config.bert_model)
bert = BertModel.from_pretrained(config.bert_model)

sentence="蚀变,矿种,矿床,断裂,地球化学元素,成因类型,地球物理特征,矿床品位,岩浆岩,变质岩,矿物," \
         "成矿单元,矿体,遥感特征,沉积岩,矿床规模,空间位置,地层,构造变形,大地构造单元,时间,矿物分带"
print(len(sentence)) #96

tokens = tokenizer.encode_plus(sentence,return_tensors='pt')#98

outputs = bert(**tokens)
sequence_output = outputs[0]
print(sequence_output.shape)
print(sequence_output.size())
torch.save(sequence_output,config.label_embedding_dir)



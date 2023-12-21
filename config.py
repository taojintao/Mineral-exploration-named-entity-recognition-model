import os
import torch
import argparse


data_dir = os.getcwd() + '/data/geodata/'
train_dir = data_dir + 'train.npz'
dev_dir = data_dir + 'dev.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'dev', 'test']
bert_model = 'pretrained_bert_models/bert-base-chinese/'
model_dir = os.getcwd() + '/experiments/geodata/'
log_dir = model_dir + 'train.log'
label_embedding_dir=data_dir + 'labels_encode.pth'
case_dir = os.getcwd() + '/case/bad_case.txt'


# load the pre-trained NER model
load_before = False

# fine-tuning the entire BERT model
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

# maximum input sentence length
max_seq_len = 300

batch_size = 16
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

seed=42

gpu = '0'

if gpu != '':
	device = torch.device(f"cuda:{gpu}")
else:
	device = torch.device("cpu")



labels = ['ALTE', 'CATE', 'DEPO', 'FAUL', 'GCA',
		  'GENE', 'GPA', 'GRA', 'MAG', 'META','MINE',
		  'MINU', 'OREB', 'RSA', 'SEDI', 'SIZE',
		  'SPAP', 'STRA', 'TECD', 'TECU', 'TIME', 'ZONA']

label2id={
	'O':0,
	'B-ALTE':1,
	'B-CATE':2,
	'B-DEPO':3,
	'B-FAUL':4,
	'B-GCA':5,
	'B-GENE':6,
	'B-GPA':7,
	'B-GRA':8,
	'B-MAG':9,
	'B-META':10,
	'B-MINE':11,
	'B-MINU':12,
	'B-OREB':13,
	'B-RSA':14,
	'B-SEDI':15,
	'B-SIZE':16,
	'B-SPAP':17,
	'B-STRA':18,
	'B-TECD':19,
	'B-TECU':20,
	'B-TIME':21,
	'B-ZONA':22,
	'I-ALTE':23,
	'I-CATE':24,
	'I-DEPO':25,
	'I-FAUL':26,
	'I-GCA':27,
	'I-GENE':28,
	'I-GPA':29,
	'I-GRA':30,
	'I-MAG':31,
	'I-META':32,
	'I-MINE':33,
	'I-MINU':34,
	'I-OREB':35,
	'I-RSA':36,
	'I-SEDI':37,
	'I-SIZE':38,
	'I-SPAP':39,
	'I-STRA':40,
	'I-TECD':41,
	'I-TECU':42,
	'I-TIME':43,
	'I-ZONA':44
}

id2label = {_id: _label for _label, _id in list(label2id.items())}

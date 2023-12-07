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


# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

# 输入句子最大长度
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

'''
def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
	parser.add_argument("--output_path", type=str, default='output/', help='模型与预处理数据的存放位置')
	parser.add_argument('--loss_type', default='CrossEntropyLoss', type=str, choices=['LabelSmoothingCrossEntropy', 'FocalLoss', 'CrossEntropyLoss'],
						help='softmax作为分类器时的损失函数类型')
	parser.add_argument("--lr", type=float, default=3e-5, help='Bert的学习率')
	parser.add_argument("--weight_decay", default=0.01, type=float, help="AdamW中WeightDecay实际上就是L2-Regularization")
	parser.add_argument('--eps', default=1.0e-08, type=float, required=False, help='AdamW优化器的衰减率')
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch_size_train", type=int, default=128)
	parser.add_argument("--batch_size_eval", type=int, default=128)
	parser.add_argument("--eval_step", type=int, default=10, help="训练多少步，查看验证集的指标")
	parser.add_argument("--max_seq_len", type=int, default=128, help="输入的最大长度")
	parser.add_argument("--lstm_hidden_dim", type=int, default=768, help="BiLSTM隐藏层维度")
	parser.add_argument("--lstm_num_layers", type=int, default=2, help="BiLSTM隐藏层数目")

	#数据集存放路径
	parser.add_argument("--train_dir", type=str, default="data/geodata_new/train.npz", help='数据集存放路径')
	parser.add_argument("--dev_dir", type=str, default="data/geodata_new/dev.npz'", help='数据集存放路径')
	parser.add_argument("--test_dir", type=str, default="data/geodata_new/test.npz", help='数据集存放路径')
	
	#预训练模型存放位置
	parser.add_argument("--bert_model", type=str, default="pretrain_model/pretrained_bert_models/bert-base-chinese/",help='预训练模型存放位置')
    parser.add_argument("--roberta_model", type=str, default="pretrained_bert_models/chinese_roberta_wwm_large_ext/",help='预训练模型存放位置')


	parser.add_argument('--grad_acc_step', default=1, type=int, required=False, help='梯度积累的步数')
	parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪阈值')
	parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
	parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
	parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.')
	
	args = parser.parse_args()
	return args
'''


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

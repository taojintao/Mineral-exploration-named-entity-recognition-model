# An entity type enhanced multi-feature fusion model for Chinese mineral exploration named entity recognition


## Dataset
使用了自己创建的中文锂矿命名实体识别标注语料, train.bio、dev.bio、test.bio文件，
每个文件有两列，格式如下：
```
磷 B-MINE
灰 I-MINE
石 I-MINE
与 O
钠 B-MINE
长 I-MINE
石 I-MINE
、 O
石 B-MINE
榴 I-MINE
石 I-MINE
和 O
白 B-MINE
云 I-MINE
母 I-MINE
共 O
生 O
或 O
伴 O
生 O
```



## Pretrained Model
需要提前下载bert-base-chinese预训练模型，将
- pytorch_model.bin
- vocab.txt

两个文件放置在./pretrained_bert_models/bert-base-chinese文件夹下


## Requirements
- python==3.8.13
- numpy==1.19.3
- pytorch==1.9.1
- pytorch-crf==0.7.2
- tqdm==4.64.1
- transformers==3.1.0

## Usage
1. 运行encode_labels.py生成实体类型句子的编码文件；
2. 运行run.py进行训练和测试。

## References
部分的代码编写参考了
- [CLUENER](https://github.com/CLUEbenchmark/CLUENER2020) 
- [LEBERT-NER-Chinese](https://github.com/yangjianxin1/LEBERT-NER-Chinese) 




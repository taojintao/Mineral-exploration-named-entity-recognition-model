# An entity type enhanced multi-feature fusion model for Chinese mineral exploration named entity recognition


## Dataset
We used a Chinese lithium deposit named entity recognition corpus created by ourselves, which includes train.bio, dev.bio, and test.bio files. Each file consists of two columns in the following format:
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
You need to download the pre-trained model bert-base-chinese，and place
- pytorch_model.bin
- vocab.txt

these two files in ./pretrained_bert_models/bert-base-chinese.


## Requirements
- python==3.8.13
- numpy==1.19.3
- pytorch==1.9.1
- pytorch-crf==0.7.2
- tqdm==4.64.1
- transformers==3.1.0

## Usage
1. Run encode_labels.py to generate the encoding file for sentences with entity types.
2. Run run.py for training and testing.

## References
Partial code implementation was referenced from
- [CLUENER](https://github.com/CLUEbenchmark/CLUENER2020) 
- [LEBERT-NER-Chinese](https://github.com/yangjianxin1/LEBERT-NER-Chinese) 




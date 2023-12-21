import torch
import random
import os
import utils
import config
import logging
import numpy as np
from data_process import Processor
from data_loader import NERDataset
from model import BertNER
from train import train, evaluate

from torch.utils.data import DataLoader
from transformers.optimization import AdamW,get_linear_schedule_with_warmup

import warnings

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    """
    Set the seed for the entire development environment.
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def test():
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    test_dataset = NERDataset(word_test, label_test, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    label_embedding=torch.load(config.label_embedding_dir).to(config.device)
    #label_embedding_bt=label_embedding.repeat_interleave(repeats=config.batch_size, dim=0)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    val_metrics = evaluate(test_loader, model, label_embedding,mode='test')
    val_precision = val_metrics['precision']
    val_recall = val_metrics['recall']
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, precision: {}, recall: {}, f1 score: {}".format(val_metrics['loss'],
                                                                                 val_precision, val_recall, val_f1))
    val_metr_labels = val_metrics['metr_labels']
    for label in config.labels:
        val_metr_label=val_metr_labels[label]
        logging.info("metrics of {}: precision: {}, recall: {}, f1 score: {}".format(label,
                                                                                     val_metr_label[0],val_metr_label[1],val_metr_label[2]))


def load_data():
    train_data = np.load(config.train_dir, allow_pickle=True)
    dev_data = np.load(config.dev_dir, allow_pickle=True) #dev
    word_train = train_data["words"]
    label_train = train_data["labels"]
    word_dev = dev_data["words"]
    label_dev = dev_data["labels"]

    return word_train, word_dev, label_train, label_dev


def run():
    """train the model"""
    # set the logger
    utils.set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))
    # process the data, separating the text and labels
    processor = Processor(config)
    processor.process()
    logging.info("--------Process Done!--------")
    # load train set and dev set
    word_train, word_dev, label_train, label_dev = load_data()
    # build dataset
    train_dataset = NERDataset(word_train, label_train, config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------Dataset Build!--------")
    # get dataset size
    train_size = len(train_dataset)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    label_embedding=torch.load(config.label_embedding_dir).to(config.device)
    #label_embedding_bt=label_embedding.repeat_interleave(repeats=config.batch_size, dim=0)

    logging.info("--------Get Dataloader!--------")
    # Prepare model
    device = config.device
    model = BertNER.from_pretrained(config.bert_model, num_labels=len(config.label2id))
    model.to(device)
    # Prepare optimizer
    if config.full_fine_tuning:
        # model.named_parameters(): [bert, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        attention_optimizer= list(model.multiheadAttn.named_parameters())
        lstm_optimizer = list(model.bilstm.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            #bert_optimizer
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            #lstm_optimizer
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            # #attention_optimizer
            # {'params': [p for n, p in attention_optimizer if not any(nd in n for nd in no_decay)],
            #  'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            # {'params': [p for n, p in attention_optimizer if any(nd in n for nd in no_decay)],
            #  'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            #classifier_optimizer
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            #crf_optimizer
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 100}  #由5改为100
        ]
            # #lstm_optimizer
            # {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
            #  'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            # {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
            #  'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            # #lstm_optimizer
            # {'params': model.bilstm.parameters(), 'lr': config.learning_rate * 5},
            # #attention_optimizer
            # {'params': [p for n, p in attention_optimizer if not any(nd in n for nd in no_decay)],
            #  'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            # {'params': [p for n, p in attention_optimizer if any(nd in n for nd in no_decay)],
            #  'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            # #attention_optimizer
            # {'params': model.multiheadAttn.parameters(), 'lr': config.learning_rate},

    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                    num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                    num_training_steps=config.epoch_num * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir,label_embedding)


if __name__ == '__main__':
    seed_everything(config.seed)
    run()
    test()

import torch
import logging
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from model import BertNER
from metrics import f1_score, bad_case
from transformers import BertTokenizer


def train_epoch(train_loader, model, optimizer, scheduler, epoch, writer,label_embedding):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask
        label_embedding_bt=label_embedding.repeat_interleave(repeats=batch_data.shape[0], dim=0)
        # compute model output and loss
        loss = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels,label_embedding=label_embedding_bt)[0]
        train_losses += loss.item()
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward(retain_graph=True)
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))
    writer.add_scalar('train loss', train_loss, epoch)


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir,label_embedding):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertNER.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    writer = SummaryWriter(model_dir)
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch, writer,label_embedding)
        val_metrics = evaluate(dev_loader, model, label_embedding, mode='dev')
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        val_f1 = val_metrics['f1']
        logging.info("Epoch: {}, dev loss: {}, precision: {}, recall: {}, f1 score: {}".format(epoch, val_metrics['loss'],
                                                                                               val_precision, val_recall, val_f1))
        writer.add_scalar('dev loss', val_metrics['loss'], epoch)
        writer.add_scalar('dev precision', val_precision, epoch)
        writer.add_scalar('dev recall', val_recall, epoch)
        writer.add_scalar('dev f1', val_f1, epoch)
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_val_f1))
            break
    logging.info("Training Finished!")


def evaluate(dev_loader, model, label_embedding, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            label_embedding_bt=label_embedding.repeat_interleave(repeats=batch_data.shape[0], dim=0)
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags,label_embedding=label_embedding_bt)[0]
            dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks,label_embedding=label_embedding_bt)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)
    if mode == 'test':
        assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        p,r,f1 = f1_score(true_tags, pred_tags, mode)
        metrics['precision'] = p
        metrics['recall'] = r
        metrics['f1'] = f1
    else:
        bad_case(true_tags, pred_tags, sent_data)
        metr_labels, p,r,f1 = f1_score(true_tags, pred_tags, mode)
        metrics['metr_labels'] = metr_labels
        metrics['precision'] = p
        metrics['recall'] = r
        metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics




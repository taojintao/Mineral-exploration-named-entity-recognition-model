import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
from metrics import get_entity_bio


class NERDataset(Dataset):
    def __init__(self, words, labels, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model)
        self.label2id = config.label2id
        self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}
        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config.device
        self.max_seq_len = config.max_seq_len

    def preprocess(self, origin_sentences, origin_labels):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples:
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:([101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            label:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        """
        data = []
        sentences = []
        labels = []
        starts = []
        ends = []
        for line in origin_sentences:
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            words = []
            word_lens = []
            for token in line:
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            # add [CLS]
            words = ['[CLS]'] + [item for token in words for item in token]
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))
        for tag in origin_labels:
            start=[0]*len(tag)
            end=[0]*len(tag)
            label_id = [self.label2id.get(t) for t in tag]
            labels.append(label_id)
            for _,start_index,end_index in get_entity_bio(tag):
                start[start_index]=1
                end[end_index]=1
            starts.append(start)
            ends.append(end)

        for sentence, label,start,end in zip(sentences, labels, starts, ends):
            data.append((sentence, label, start, end))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        start = self.dataset[idx][2]
        end = self.dataset[idx][3]
        return [word, label, start,end]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: Pad each batch of data to the same length (the length of the longest data in the batch, or you can use a specified max_seq_len)
            2. aligning: identify the positions in each sentence sequence where there are label items, ensuring alignment between text and labels.
            3. tensor：convert data to tensor
        """
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        starts = [x[2] for x in batch]
        ends = [x[3] for x in batch]

        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        #max_len = max([len(s[0]) for s in sentences])

        # set length of longest sentence
        max_len = self.max_seq_len

        max_label_len = 0

        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts = []

        # initialize padding data
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            if cur_len > max_len:
                batch_data[j][:] = sentences[j][0][:max_len]
            else:
                batch_data[j][:cur_len] = sentences[j][0]
            # find the index of data with labels excluding [CLS]
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # truncating, padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        batch_starts = np.zeros((batch_len, max_label_len))
        batch_ends = np.zeros((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            if cur_tags_len > max_label_len:
                batch_labels[j][:] = labels[j][:max_label_len]
                batch_starts[j][:] = starts[j][:max_label_len]
                batch_ends[j][:] = ends[j][:max_label_len]
            else:
                batch_labels[j][:cur_tags_len] = labels[j]
                batch_starts[j][:cur_tags_len] = starts[j]
                batch_ends[j][:cur_tags_len] = ends[j]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        batch_starts = torch.tensor(batch_starts,dtype=torch.long)
        batch_ends = torch.tensor(batch_ends,dtype=torch.long)


        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch_starts = batch_starts.to(self.device)
        batch_ends = batch_ends.to(self.device)
        return [batch_data, batch_label_starts, batch_labels, batch_starts, batch_ends]




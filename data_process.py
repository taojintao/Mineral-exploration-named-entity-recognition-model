import os
import logging
import numpy as np
from tqdm import trange

class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config

    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files:
            self.preprocess(file_name)

    def preprocess(self, mode):
        """
        params:
            separate the text and labels in each line of the BIO file, and store them as words and labels lists
        examples:
            words：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
            labels：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        input_dir = self.data_dir + str(mode) + '.bio'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []

        with open(input_dir, 'r', encoding='utf8') as f:
            lines = f.readlines()
            words = []
            labels = []
            for idx in trange(len(lines)):
                line = lines[idx].rstrip()
                if not line:
                    assert len(words) == len(labels), (len(words), len(labels))
                    word_list.append(words)
                    label_list.append(labels)
                    words = []
                    labels = []
                else:
                    word, label = line.split()
                    words.append(word)
                    labels.append(label)
        # save as a binary file
        np.savez_compressed(output_dir, words=word_list, labels=label_list)
        logging.info("--------{} data process DONE!--------".format(mode))


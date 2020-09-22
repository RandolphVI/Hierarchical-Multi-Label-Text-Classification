# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import math
import logging
import heapq
import json
import gensim
import torch
import numpy as np
import torch.nn.utils.rnn as rnn_utils

from texttable import Texttable
from gensim.models import KeyedVectors
from gensim.models import word2vec


def option():
    """
    Choose training or restore pattern.

    Returns:
        The OPTION
    """
    OPTION = input("[Input] Train or Restore? (T/R): ")
    while not (OPTION.upper() in ['T', 'R']):
        OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger
        input_file: The logger file path
        level: The logger level
    Returns:
        The logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(input_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    return logger


def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.warning('\n' + t.draw())


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted onehot labels based on the topK number.
    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def create_prediction_file(save_dir, identifiers, predictions):
    """
    Create the prediction file.

    Args:
        save_dir: The all classes predicted results provided by network
        identifiers: The data record id
        predictions: The predict scores
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    preds_file = os.path.abspath(os.path.join(save_dir, 'submission.json'))
    with open(preds_file, 'w') as fout:
        tmp_dict = {}
        for index, predicted_label in enumerate(predictions):
            if identifiers[index] not in tmp_dict:
                tmp_dict[identifiers[index]] = [predicted_label]
            else:
                tmp_dict[identifiers[index]].append(predicted_label)

        for key in tmp_dict.keys():
            data_record = {
                'item_id': key,
                'label_list': tmp_dict[key],
            }
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def load_word2vec_matrix(word2vec_file):
    """
    Return the word2vec model matrix.

    Args:
        word2vec_file: The word2vec file
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    model = gensim.models.Word2Vec.load(word2vec_file)
    vocab_size = model.wv.vectors.shape[0]
    embedding_size = model.vector_size
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():
        if key is not None:
            embedding_matrix[value] = model[key]
    return vocab_size, embedding_size, embedding_matrix


def load_data_and_labels(args, input_file):
    """
    Load research data from files, splits the data into words and generates labels.
    Return the dict <Data> (includes the record tokenindex and record labels).

    Args:
        args: The arguments.
        input_file: The research record.
    Returns:
        The dict <Data> (includes the record tokenindex and record labels)
    Raises:
        IOError: If word2vec word2vec_model file doesn't exist
    """
    # Load word2vec file
    if not os.path.isfile(args.word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    if not input_file.endswith('.json'):
        raise IOError("[Error] The research record is not a json file. "
                      "Please preprocess the research record into the json file.")

    word2vec_model = gensim.models.Word2Vec.load(args.word2vec_file)
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def _token_to_index(x):
        result = []
        for item in x:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    def _create_onehot_labels(labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label

    Data = dict()
    with open(input_file) as fin:
        Data['id'] = []
        Data['content'] = []
        Data['section'] = []
        Data['subsection'] = []
        Data['group'] = []
        Data['subgroup'] = []
        Data['onehot_labels'] = []
        Data['labels'] = []

        for eachline in fin:
            record = json.loads(eachline)
            id = record['id']
            content = record['abstract']
            section = record['section']
            subsection = record['subsection']
            group = record['group']
            subgroup = record['subgroup']
            labels = record['labels']

            Data['id'].append(id)
            Data['content'].append(_token_to_index(content))
            Data['section'].append(_create_onehot_labels(section, args.num_classes_list[0]))
            Data['subsection'].append(_create_onehot_labels(subsection, args.num_classes_list[1]))
            Data['group'].append(_create_onehot_labels(group, args.num_classes_list[2]))
            Data['subgroup'].append(_create_onehot_labels(subgroup, args.num_classes_list[3]))
            Data['onehot_labels'].append(_create_onehot_labels(labels, args.total_classes))
            Data['labels'].append(labels)

    return Data


class MyData(torch.utils.data.Dataset):
    """
    Define the IterableDataset.
    """
    def __init__(self, data: dict):
        self.seqs = data['content']
        self.section = data['section']
        self.subsection = data['subsection']
        self.group = data['group']
        self.subgroup = data['subgroup']
        self.onehot_labels = data['onehot_labels']

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.section[idx], self.subsection[idx], \
               self.group[idx], self.subgroup[idx], self.onehot_labels[idx]


def collate_fn(data):
    """
    Version for PyTorch

    Args:
        data: The research data. 0-dim: word token index / 1-dim: data label
    Returns:
        pad_content: The padded data
        lens: The ground truth lengths
        labels: The data labels
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(i[0]) for i in data]
    x = [torch.tensor(i[0]) for i in data]
    section = torch.FloatTensor([i[1] for i in data])
    subsection = torch.FloatTensor([i[2] for i in data])
    group = torch.FloatTensor([i[3] for i in data])
    subgroup = torch.FloatTensor([i[4] for i in data])
    y = torch.FloatTensor([i[5] for i in data])
    pad_content = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0.)
    return pad_content.unsqueeze(-1), lens, section, subsection, group, subgroup, y
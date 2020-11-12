# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import time
import heapq
import gensim
import logging
import json
import numpy as np
from collections import OrderedDict
from pylab import *
from texttable import Texttable
from gensim.models import KeyedVectors
from tflearn.data_utils import pad_sequences


def _option(pattern):
    """
    Get the option according to the pattern.
    pattern 0: Choose training or restore.
    pattern 1: Choose best or latest checkpoint.

    Args:
        pattern: 0 for training step. 1 for testing step.
    Returns:
        The OPTION.
    """
    if pattern == 0:
        OPTION = input("[Input] Train or Restore? (T/R): ")
        while not (OPTION.upper() in ['T', 'R']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    if pattern == 1:
        OPTION = input("Load Best or Latest Model? (B/L): ")
        while not (OPTION.isalpha() and OPTION.upper() in ['B', 'L']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger.
        input_file: The logger file path.
        level: The logger level.
    Returns:
        The logger.
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
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.info('\n' + t.draw())


def get_out_dir(option, logger):
    """
    Get the out dir for saving model checkpoints.

    Args:
        option: Train or Restore.
        logger: The logger.
    Returns:
        The output dir for model checkpoints.
    """
    if option == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {0}\n".format(out_dir))
    if option == 'R':
        MODEL = input("[Input] Please input the checkpoints model you want to restore, "
                      "it should be like (1490175368): ")  # The model you want to restore

        while not (MODEL.isdigit() and len(MODEL) == 10):
            MODEL = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
        logger.info("Writing to {0}\n".format(out_dir))
    return out_dir


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name.
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def create_prediction_file(output_file, data_id, true_labels, predict_labels, predict_scores):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted results provided by network.
        data_id: The data record id info provided by dict <Data>.
        true_labels: The all true labels.
        predict_labels: The all predict labels by threshold.
        predict_scores: The all predict scores by threshold.
    Raises:
        IOError: If the prediction file is not a .json file.
    """
    if not output_file.endswith('.json'):
        raise IOError("[Error] The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(predict_labels)
        for i in range(data_size):
            data_record = OrderedDict([
                ('id', data_id[i]),
                ('labels', [int(i) for i in true_labels[i]]),
                ('predict_labels', [int(i) for i in predict_labels[i]]),
                ('predict_scores', [round(i, 4) for i in predict_scores[i]])
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted one-hot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.

    Args:
        scores: The all classes predicted scores provided by network.
        threshold: The threshold (default: 0.5).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
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
    Get the predicted one-hot labels based on the topK.

    Args:
        scores: The all classes predicted scores provided by network.
        top_num: The max topK number (default: 5).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
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


def get_label_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network.
        threshold: The threshold (default: 0.5).
    Returns:
        predicted_labels: The predicted labels.
        predicted_scores: The predicted scores.
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        score_list = []
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                index_list.append(index)
                score_list.append(predict_score)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            score_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def get_label_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network.
        top_num: The max topK number (default: 5).
    Returns:
        The predicted labels.
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def create_metadata_file(word2vec_file, output_file):
    """
    Create the metadata file based on the corpus file (Used for the Embedding Visualization later).

    Args:
        word2vec_file: The word2vec file.
        output_file: The metadata file path.
    Raises:
        IOError: If word2vec model file doesn't exist.
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist.")

    wv = KeyedVectors.load(word2vec_file, mmap='r')
    word2idx = dict([(k, v.index) for k, v in wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("[Warning] Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def load_word2vec_matrix(word2vec_file):
    """
    Get the word2idx dict and embedding matrix.

    Args:
        word2vec_file: The word2vec file.
    Returns:
        word2idx: The word2idx dict.
        embedding_matrix: The word2vec model matrix.
    Raises:
        IOError: If word2vec model file doesn't exist.
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    wv = KeyedVectors.load(word2vec_file, mmap='r')

    word2idx = OrderedDict({"_UNK": 0})
    embedding_size = wv.vector_size
    for k, v in wv.vocab.items():
        word2idx[k] = v.index + 1
    vocab_size = len(word2idx)

    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in word2idx.items():
        if key == "_UNK":
            embedding_matrix[value] = [0. for _ in range(embedding_size)]
        else:
            embedding_matrix[value] = wv[key]
    return word2idx, embedding_matrix


def load_data_and_labels(args, input_file, word2idx: dict):
    """
    Load research data from files, padding sentences and generate one-hot labels.

    Args:
        args: The arguments.
        input_file: The research record.
        word2idx: The word2idx dict.
    Returns:
        The dict <Data> (includes the record tokenindex and record labels)
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    if not input_file.endswith('.json'):
        raise IOError("[Error] The research record is not a json file. "
                      "Please preprocess the research record into the json file.")

    def _token_to_index(x: list):
        result = []
        for item in x:
            if item not in word2idx.keys():
                result.append(word2idx['_UNK'])
            else:
                word_idx = word2idx[item]
                result.append(word_idx)
        return result

    def _create_onehot_labels(labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label

    Data = dict()
    with open(input_file) as fin:
        Data['id'] = []
        Data['content_index'] = []
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
            Data['content_index'].append(_token_to_index(content))
            Data['content'].append(content)
            Data['section'].append(_create_onehot_labels(section, args.num_classes_list[0]))
            Data['subsection'].append(_create_onehot_labels(subsection, args.num_classes_list[1]))
            Data['group'].append(_create_onehot_labels(group, args.num_classes_list[2]))
            Data['subgroup'].append(_create_onehot_labels(subgroup, args.num_classes_list[3]))
            Data['onehot_labels'].append(_create_onehot_labels(labels, args.total_classes))
            Data['labels'].append(labels)
        Data['pad_seqs'] = pad_sequences(Data['content_index'], maxlen=args.pad_seq_len, value=0.)
    return Data


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data.
        batch_size: The size of the data batch.
        num_epochs: The number of epochs.
        shuffle: Shuffle or not (default: True).
    Returns:
        A batch iterator for data set.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

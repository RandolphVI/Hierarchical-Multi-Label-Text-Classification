# -*- coding:utf-8 -*-
__author__ = 'randolph'

import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn

sys.path.append('../')

from layers import HmcNet
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from tqdm import tqdm, trange
import torch.nn.utils.rnn as rnn_utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


args = parser.parameter_parser()
OPTION = dh.option()
logger = dh.logger_fn("ptlog", "logs/{0}-{1}.log".format('Train' if OPTION == 'T' else 'Restore', time.asctime()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_input_data(record):
    """
    Feeding the features and targets into the Device (GPU if possible).
    """
    x, xlens, y_section, y_subsec, y_group, y_subgroup, y = record
    batch_x_pack = rnn_utils.pack_padded_sequence(x, xlens, batch_first=True)
    return batch_x_pack.to(device), xlens, y_section.to(device), y_subsec.to(device), \
           y_group.to(device), y_subgroup.to(device), y.to(device)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')


def print_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            print("weight", m.weight.data)
            print("bias:", m.bias.data)
            print("next...")


def train():
    """Training QuesNet model."""
    dh.tab_printer(args, logger)

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")
    logger.info("Data processing...")
    train_dataset = dh.load_data_and_labels(args, args.train_file)
    val_dataset = dh.load_data_and_labels(args, args.validation_file)

    logger.info("Data padding...")
    train_dataset = dh.MyData(train_dataset)
    val_dataset = dh.MyData(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dh.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dh.collate_fn)

    # Load word2vec model
    VOCAB_SIZE, EMBEDDING_SIZE, pretrained_word2vec_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Init network
    logger.info("Init nn...")
    net = HmcNet(args, VOCAB_SIZE, EMBEDDING_SIZE, pretrained_word2vec_matrix).to(device)

    # weights_init(model=net)
    # print_weight(model=net)

    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)

    if OPTION == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        saver = cm.BestCheckpointSaver(save_dir=out_dir, num_to_keep=args.num_checkpoints, maximize=False)
        logger.info("Writing to {0}\n".format(out_dir))
    elif OPTION == 'R':
        timestamp = input("[Input] Please input the checkpoints model you want to restore: ")
        while not (timestamp.isdigit() and len(timestamp) == 10):
            timestamp = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        saver = cm.BestCheckpointSaver(save_dir=out_dir, num_to_keep=args.num_checkpoints, maximize=False)
        logger.info("Writing to {0}\n".format(out_dir))
        checkpoint = torch.load(out_dir)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info("Training...")
    writer = SummaryWriter('summary')

    def eval_model(val_loader, epoch):
        """
        Evaluate on the validation set.
        """
        net.eval()
        eval_loss = 0.0
        eval_pre_tk = [0.0 for _ in range(args.topK)]
        eval_rec_tk = [0.0 for _ in range(args.topK)]
        eval_F_tk = [0.0 for _ in range(args.topK)]
        true_onehot_labels = []
        predicted_onehot_scores = []
        predicted_onehot_labels_ts = []
        predicted_onehot_labels_tk = [[] for _ in range(args.topK)]
        for batch_idx, batch in enumerate(val_loader):
            record = create_input_data(batch)
            logits, scores, avg_batch_loss = net(record)
            eval_loss += avg_batch_loss.item()

            y_val = record[-1]
            # Prepare for calculating metrics
            for onehot_labels in y_val:
                true_onehot_labels.append(onehot_labels.tolist())
            for onehot_scores in scores:
                predicted_onehot_scores.append(onehot_scores.tolist())
            # Predict by threshold
            batch_predicted_onehot_labels_ts = \
                dh.get_onehot_label_threshold(scores=scores.cpu().detach().numpy(), threshold=args.threshold)
            for onehot_labels in batch_predicted_onehot_labels_ts:
                predicted_onehot_labels_ts.append(onehot_labels)
            # Predict by topK
            for num in range(args.topK):
                batch_predicted_onehot_labels_tk = \
                    dh.get_onehot_label_topk(scores=scores.cpu().detach().numpy(), top_num=num + 1)
                for onehot_labels in batch_predicted_onehot_labels_tk:
                    predicted_onehot_labels_tk[num].append(onehot_labels)

        batch_cnt = batch_idx + 1
        # Calculate Precision & Recall & F1 (Threshold)
        eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                      y_pred=np.array(predicted_onehot_labels_ts), average='micro')
        eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                   y_pred=np.array(predicted_onehot_labels_ts), average='micro')
        eval_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                             y_pred=np.array(predicted_onehot_labels_ts), average='micro')

        # Calculate Precision & Recall & F1 (TopK)
        for num in range(args.topK):
            eval_pre_tk[num] = precision_score(y_true=np.array(true_onehot_labels),
                                               y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')
            eval_rec_tk[num] = recall_score(y_true=np.array(true_onehot_labels),
                                            y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')
            eval_F_tk[num] = f1_score(y_true=np.array(true_onehot_labels),
                                      y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')

        # Calculate the average AUC
        eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                 y_score=np.array(predicted_onehot_scores), average='micro')
        # Calculate the average PR
        eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                           y_score=np.array(predicted_onehot_scores), average='micro')

        logger.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                    .format(eval_loss / batch_cnt, eval_auc, eval_prc))
        logger.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}"
                    .format(eval_pre_ts, eval_rec_ts, eval_F_ts))
        logger.info("Predict by topK:")
        for num in range(args.topK):
            logger.info("Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}"
                        .format(num + 1, eval_pre_tk[num], eval_rec_tk[num], eval_F_tk[num]))

        cur_value = eval_prc
        writer.add_scalar('validation loss', eval_loss, epoch)
        writer.add_scalar('validation PRECISION', eval_pre_ts, epoch)
        writer.add_scalar('validation RECALL', eval_rec_ts, epoch)
        writer.add_scalar('validation F1', eval_F_ts, epoch)
        writer.add_scalar('validation AUC', eval_auc, epoch)
        writer.add_scalar('validation PRC', eval_prc, epoch)
        return cur_value

    for epoch in tqdm(range(args.epochs), desc="Epochs:", leave=True):
        # Training step
        batches = trange(len(train_loader), desc="Batches", leave=True)
        for batch_cnt, batch in zip(batches, train_loader):
            net.train()
            record = create_input_data(batch)
            optimizer.zero_grad()   # 如果不置零，Variable 的梯度在每次 backward 的时候都会累加
            logits, scores, avg_batch_loss = net(record)
            # TODO
            avg_batch_loss.backward()
            optimizer.step()    # Parameter updating
            batches.set_description("Batches (Loss={:.4f})".format(avg_batch_loss.item()))
            logger.info('[epoch {0}, batch {1}] loss: {2:.4f}'.format(epoch + 1, batch_cnt, avg_batch_loss.item()))
            writer.add_scalar('training loss', avg_batch_loss, batch_cnt)
        # Evaluation step
        cur_value = eval_model(val_loader, epoch)
        saver.handle(cur_value, net, optimizer, epoch)
    writer.close()

    logger.info('Training Finished.')


if __name__ == "__main__":
    train()

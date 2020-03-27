# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging

sys.path.append('../')
logging.getLogger('tensorflow').disabled = True

import numpy as np
import tensorflow as tf
from text_harnn import TextHARNN
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

args = parser.parameter_parser()
OPTION = dh._option(pattern=0)
logger = dh.logger_fn("tflog", "logs/{0}-{1}.log".format('Train' if OPTION == 'T' else 'Restore', time.asctime()))


def train_harnn():
    """Training HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")
    logger.info("Data processing...")
    train_data = dh.load_data_and_labels(args.train_file, args.num_classes_list, args.total_classes,
                                         args.word2vec_file, data_aug_flag=False)
    val_data = dh.load_data_and_labels(args.validation_file, args.num_classes_list, args.total_classes,
                                       args.word2vec_file, data_aug_flag=False)

    logger.info("Data padding...")
    x_train, y_train, y_train_tuple = dh.pad_data(train_data, args.pad_seq_len)
    x_val, y_val, y_val_tuple = dh.pad_data(val_data, args.pad_seq_len)

    # Build vocabulary
    VOCAB_SIZE, pretrained_word2vec_matrix = dh.load_word2vec_matrix(args.embedding_dim, args.word2vec_file)

    # Build a graph and harnn object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            harnn = TextHARNN(
                sequence_length=args.pad_seq_len,
                vocab_size=VOCAB_SIZE,
                embedding_type=args.embedding_type,
                embedding_size=args.embedding_dim,
                lstm_hidden_size=args.lstm_dim,
                attention_unit_size=args.attention_dim,
                fc_hidden_size=args.fc_dim,
                num_classes_list=args.num_classes_list,
                total_classes=args.total_classes,
                batch_size=args.batch_size,
                l2_reg_lambda=args.l2_lambda,
                alpha=args.alpha,
                pretrained_embedding=pretrained_word2vec_matrix)

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                           global_step=harnn.global_step, decay_steps=args.decay_steps,
                                                           decay_rate=args.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(harnn.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=args.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=harnn.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            out_dir = dh.get_out_dir(OPTION, logger)
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", harnn.loss)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            if OPTION == 'R':
                # Load harnn model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            if OPTION == 'T':
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(harnn.global_step)

            def train_step(x_batch, y_batch, y_batch_tuple):
                """A single training step"""
                y_batch_first = [i[0] for i in y_batch_tuple]
                y_batch_second = [j[1] for j in y_batch_tuple]
                y_batch_third = [k[2] for k in y_batch_tuple]
                y_batch_fourth = [t[3] for t in y_batch_tuple]

                feed_dict = {
                    harnn.input_x: x_batch,
                    harnn.input_y_first: y_batch_first,
                    harnn.input_y_second: y_batch_second,
                    harnn.input_y_third: y_batch_third,
                    harnn.input_y_fourth: y_batch_fourth,
                    harnn.input_y: y_batch,
                    harnn.dropout_keep_prob: args.dropout_rate,
                    harnn.beta: args.beta,
                    harnn.is_training: True
                }
                _, step, summaries, loss = sess.run(
                    [train_op, harnn.global_step, train_summary_op, harnn.loss], feed_dict)
                logger.info("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_val, y_val, y_val_tuple, writer=None):
                """Evaluates model on a validation set"""
                batches_validation = dh.batch_iter(
                    list(zip(x_val, y_val, y_val_tuple)), args.batch_size, 1)

                # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
                eval_counter, eval_loss = 0, 0.0

                eval_pre_tk = [0.0] * args.topK
                eval_rec_tk = [0.0] * args.topK
                eval_F_tk = [0.0] * args.topK

                true_onehot_labels = []
                predicted_onehot_scores = []
                predicted_onehot_labels_ts = []
                predicted_onehot_labels_tk = [[] for _ in range(args.topK)]

                for batch_validation in batches_validation:
                    x_batch_val, y_batch_val, y_batch_val_tuple = zip(*batch_validation)

                    y_batch_val_first = [i[0] for i in y_batch_val_tuple]
                    y_batch_val_second = [j[1] for j in y_batch_val_tuple]
                    y_batch_val_third = [k[2] for k in y_batch_val_tuple]
                    y_batch_val_fourth = [t[3] for t in y_batch_val_tuple]

                    feed_dict = {
                        harnn.input_x: x_batch_val,
                        harnn.input_y_first: y_batch_val_first,
                        harnn.input_y_second: y_batch_val_second,
                        harnn.input_y_third: y_batch_val_third,
                        harnn.input_y_fourth: y_batch_val_fourth,
                        harnn.input_y: y_batch_val,
                        harnn.dropout_keep_prob: 1.0,
                        harnn.beta: args.beta,
                        harnn.is_training: False
                    }
                    step, summaries, scores, cur_loss = sess.run(
                        [harnn.global_step, validation_summary_op, harnn.scores, harnn.loss], feed_dict)

                    # Prepare for calculating metrics
                    for onehot_labels in y_batch_val:
                        true_onehot_labels.append(onehot_labels)
                    for onehot_scores in scores:
                        predicted_onehot_scores.append(onehot_scores)

                    # Predict by threshold
                    batch_predicted_onehot_labels_ts = \
                        dh.get_onehot_label_threshold(scores=scores, threshold=args.threshold)

                    for onehot_labels in batch_predicted_onehot_labels_ts:
                        predicted_onehot_labels_ts.append(onehot_labels)

                    # Predict by topK
                    for top_num in range(args.topK):
                        batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores, top_num=top_num+1)

                        for onehot_labels in batch_predicted_onehot_labels_tk:
                            predicted_onehot_labels_tk[top_num].append(onehot_labels)

                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1

                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)

                # Calculate Precision & Recall & F1
                eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                              y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                           y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                                     y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                # Calculate the average AUC
                eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                         y_score=np.array(predicted_onehot_scores), average='micro')
                # Calculate the average PR
                eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                                   y_score=np.array(predicted_onehot_scores), average='micro')

                for top_num in range(args.topK):
                    eval_pre_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                           y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                           average='micro')
                    eval_rec_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                        y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                        average='micro')
                    eval_F_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                                  y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                  average='micro')

                return eval_loss, eval_auc, eval_prc, eval_rec_ts, eval_pre_ts, eval_F_ts, \
                       eval_rec_tk, eval_pre_tk, eval_F_tk

            # Generate batches
            batches_train = dh.batch_iter(
                list(zip(x_train, y_train, y_train_tuple)), args.batch_size, args.epochs)

            num_batches_per_epoch = int((len(x_train) - 1) / args.batch_size) + 1

            # Training loop. For each batch...
            for batch_train in batches_train:
                x_batch_train, y_batch_train, y_batch_train_tuple = zip(*batch_train)
                train_step(x_batch_train, y_batch_train, y_batch_train_tuple)
                current_step = tf.train.global_step(sess, harnn.global_step)

                if current_step % args.evaluate_steps == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_auc, eval_prc, \
                    eval_rec_ts, eval_pre_ts, eval_F_ts, eval_rec_tk, eval_pre_tk, eval_F_tk = \
                        validation_step(x_val, y_val, y_val_tuple, writer=validation_summary_writer)

                    logger.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                                .format(eval_loss, eval_auc, eval_prc))

                    # Predict by threshold
                    logger.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}"
                                .format(eval_pre_ts, eval_rec_ts, eval_F_ts))

                    # Predict by topK
                    logger.info("Predict by topK:")
                    for top_num in range(args.topK):
                        logger.info("Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}"
                                    .format(top_num+1, eval_pre_tk[top_num], eval_rec_tk[top_num], eval_F_tk[top_num]))
                    best_saver.handle(eval_prc, sess, current_step)
                if current_step % args.checkpoint_steps == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))
                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("Epoch {0} has finished!".format(current_epoch))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    train_harnn()

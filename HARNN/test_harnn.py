# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging
import numpy as np

sys.path.append('../')
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

args = parser.parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("tflog", "logs/Test-{0}.log".format(time.asctime()))

CPT_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_CPT_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'output/' + MODEL


def test_harnn():
    """Test HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels(args.test_file, args.num_classes_list, args.total_classes,
                                        args.word2vec_file, data_aug_flag=False)

    logger.info("Data padding...")
    x_test, y_test, y_test_tuple = dh.pad_data(test_data, args.pad_seq_len)
    y_test_labels = test_data.labels

    # Load harnn model
    OPTION = dh._option(pattern=1)
    if OPTION == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    else:
        logger.info("Loading latest model...")
        checkpoint_file = tf.train.latest_checkpoint(CPT_DIR)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y_first = graph.get_operation_by_name("input_y_first").outputs[0]
            input_y_second = graph.get_operation_by_name("input_y_second").outputs[0]
            input_y_third = graph.get_operation_by_name("input_y_third").outputs[0]
            input_y_fourth = graph.get_operation_by_name("input_y_fourth").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            beta = graph.get_operation_by_name("beta").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            first_scores = graph.get_operation_by_name("first-output/scores").outputs[0]
            second_scores = graph.get_operation_by_name("second-output/scores").outputs[0]
            third_scores = graph.get_operation_by_name("third-output/scores").outputs[0]
            fourth_scores = graph.get_operation_by_name("fourth-output/scores").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            loss = graph.get_operation_by_name("loss/loss").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/scores|second-output/scores|third-output/scores|fourth-output/scores|output/scores"

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(zip(x_test, y_test, y_test_tuple, y_test_labels)),
                                    args.batch_size, 1, shuffle=False)

            test_counter, test_loss = 0, 0.0

            # Collect the predictions here
            true_labels = []
            predicted_labels = []
            predicted_scores = []

            # Collect for calculating metrics
            true_onehot_labels = []
            predicted_onehot_scores = []
            predicted_onehot_labels_ts = []
            predicted_onehot_labels_tk = [[] for _ in range(args.topK)]

            true_onehot_first_labels = []
            true_onehot_second_labels = []
            true_onehot_third_labels = []
            true_onehot_fourth_labels = []
            predicted_onehot_scores_first = []
            predicted_onehot_scores_second = []
            predicted_onehot_scores_third = []
            predicted_onehot_scores_fourth = []
            predicted_onehot_labels_first = []
            predicted_onehot_labels_second = []
            predicted_onehot_labels_third = []
            predicted_onehot_labels_fourth = []

            for batch_test in batches:
                x_batch_test, y_batch_test, y_batch_test_tuple, y_batch_test_labels = zip(*batch_test)

                y_batch_test_first = [i[0] for i in y_batch_test_tuple]
                y_batch_test_second = [j[1] for j in y_batch_test_tuple]
                y_batch_test_third = [k[2] for k in y_batch_test_tuple]
                y_batch_test_fourth = [t[3] for t in y_batch_test_tuple]

                feed_dict = {
                    input_x: x_batch_test,
                    input_y_first: y_batch_test_first,
                    input_y_second: y_batch_test_second,
                    input_y_third: y_batch_test_third,
                    input_y_fourth: y_batch_test_fourth,
                    input_y: y_batch_test,
                    dropout_keep_prob: 1.0,
                    beta: args.beta,
                    is_training: False
                }
                batch_first_scores, batch_second_scores, batch_third_scores, batch_fourth_scores, batch_scores, cur_loss = \
                    sess.run([first_scores, second_scores, third_scores, fourth_scores, scores, loss], feed_dict)

                # Prepare for calculating metrics
                for onehot_labels in y_batch_test:
                    true_onehot_labels.append(onehot_labels)
                for onehot_labels in y_batch_test_first:
                    true_onehot_first_labels.append(onehot_labels)
                for onehot_labels in y_batch_test_second:
                    true_onehot_second_labels.append(onehot_labels)
                for onehot_labels in y_batch_test_third:
                    true_onehot_third_labels.append(onehot_labels)
                for onehot_labels in y_batch_test_fourth:
                    true_onehot_fourth_labels.append(onehot_labels)

                for onehot_scores in batch_scores:
                    predicted_onehot_scores.append(onehot_scores)
                for onehot_scores in batch_first_scores:
                    predicted_onehot_scores_first.append(onehot_scores)
                for onehot_scores in batch_second_scores:
                    predicted_onehot_scores_second.append(onehot_scores)
                for onehot_scores in batch_third_scores:
                    predicted_onehot_scores_third.append(onehot_scores)
                for onehot_scores in batch_fourth_scores:
                    predicted_onehot_scores_fourth.append(onehot_scores)

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores, threshold=args.threshold)

                # Add results to collection
                for labels in y_batch_test_labels:
                    true_labels.append(labels)
                for labels in batch_predicted_labels_ts:
                    predicted_labels.append(labels)
                for values in batch_predicted_scores_ts:
                    predicted_scores.append(values)

                # Get one-hot prediction by threshold
                batch_predicted_onehot_labels_ts = \
                    dh.get_onehot_label_threshold(scores=batch_scores, threshold=args.threshold)
                batch_predicted_onehot_labels_first = \
                    dh.get_onehot_label_threshold(scores=batch_first_scores, threshold=args.threshold)
                batch_predicted_onehot_labels_second = \
                    dh.get_onehot_label_threshold(scores=batch_second_scores, threshold=args.threshold)
                batch_predicted_onehot_labels_third = \
                    dh.get_onehot_label_threshold(scores=batch_third_scores, threshold=args.threshold)
                batch_predicted_onehot_labels_fourth = \
                    dh.get_onehot_label_threshold(scores=batch_fourth_scores, threshold=args.threshold)

                for onehot_labels in batch_predicted_onehot_labels_ts:
                    predicted_onehot_labels_ts.append(onehot_labels)
                for onehot_labels in batch_predicted_onehot_labels_first:
                    predicted_onehot_labels_first.append(onehot_labels)
                for onehot_labels in batch_predicted_onehot_labels_second:
                    predicted_onehot_labels_second.append(onehot_labels)
                for onehot_labels in batch_predicted_onehot_labels_third:
                    predicted_onehot_labels_third.append(onehot_labels)
                for onehot_labels in batch_predicted_onehot_labels_fourth:
                    predicted_onehot_labels_fourth.append(onehot_labels)

                # Get one-hot prediction by topK
                for i in range(args.topK):
                    batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=batch_scores, top_num=i + 1)

                    for onehot_labels in batch_predicted_onehot_labels_tk:
                        predicted_onehot_labels_tk[i].append(onehot_labels)

                test_loss = test_loss + cur_loss
                test_counter = test_counter + 1

            # Calculate Precision & Recall & F1
            test_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                          y_pred=np.array(predicted_onehot_labels_ts), average='micro')

            test_pre_first = precision_score(y_true=np.array(true_onehot_first_labels),
                                             y_pred=np.array(predicted_onehot_labels_first), average='micro')
            test_pre_second = precision_score(y_true=np.array(true_onehot_second_labels),
                                              y_pred=np.array(predicted_onehot_labels_second), average='micro')
            test_pre_third = precision_score(y_true=np.array(true_onehot_third_labels),
                                             y_pred=np.array(predicted_onehot_labels_third), average='micro')
            test_pre_fourth = precision_score(y_true=np.array(true_onehot_fourth_labels),
                                              y_pred=np.array(predicted_onehot_labels_fourth), average='micro')

            test_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                       y_pred=np.array(predicted_onehot_labels_ts), average='micro')

            test_rec_first = recall_score(y_true=np.array(true_onehot_first_labels),
                                          y_pred=np.array(predicted_onehot_labels_first), average='micro')
            test_rec_second = recall_score(y_true=np.array(true_onehot_second_labels),
                                           y_pred=np.array(predicted_onehot_labels_second), average='micro')
            test_rec_third = recall_score(y_true=np.array(true_onehot_third_labels),
                                          y_pred=np.array(predicted_onehot_labels_third), average='micro')
            test_rec_fourth = recall_score(y_true=np.array(true_onehot_fourth_labels),
                                           y_pred=np.array(predicted_onehot_labels_fourth), average='micro')

            test_F1_ts = f1_score(y_true=np.array(true_onehot_labels),
                                  y_pred=np.array(predicted_onehot_labels_ts), average='micro')

            test_F1_first = f1_score(y_true=np.array(true_onehot_first_labels),
                                     y_pred=np.array(predicted_onehot_labels_first), average='micro')
            test_F1_second = f1_score(y_true=np.array(true_onehot_second_labels),
                                      y_pred=np.array(predicted_onehot_labels_second), average='micro')
            test_F1_third = f1_score(y_true=np.array(true_onehot_third_labels),
                                     y_pred=np.array(predicted_onehot_labels_third), average='micro')
            test_F1_fourth = f1_score(y_true=np.array(true_onehot_fourth_labels),
                                      y_pred=np.array(predicted_onehot_labels_fourth), average='micro')

            # Calculate the average AUC
            test_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                     y_score=np.array(predicted_onehot_scores), average='micro')

            # Calculate the average PR
            test_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                               y_score=np.array(predicted_onehot_scores), average="micro")
            test_prc_first = average_precision_score(y_true=np.array(true_onehot_first_labels),
                                                     y_score=np.array(predicted_onehot_scores_first), average="micro")
            test_prc_second = average_precision_score(y_true=np.array(true_onehot_second_labels),
                                                      y_score=np.array(predicted_onehot_scores_second), average="micro")
            test_prc_third = average_precision_score(y_true=np.array(true_onehot_third_labels),
                                                     y_score=np.array(predicted_onehot_scores_third), average="micro")
            test_prc_fourth = average_precision_score(y_true=np.array(true_onehot_fourth_labels),
                                                      y_score=np.array(predicted_onehot_scores_fourth), average="micro")

            test_loss = float(test_loss / test_counter)

            logger.info("All Test Dataset: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                        .format(test_loss, test_auc, test_prc))
            # Predict by threshold
            logger.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F1 {2:g}"
                        .format(test_pre_ts, test_rec_ts, test_F1_ts))

            logger.info("Predict by threshold in Level-1: Precision {0:g}, Recall {1:g}, F1 {2:g}, AUPRC {3:g}"
                        .format(test_pre_first, test_rec_first, test_F1_first, test_prc_first))
            logger.info("Predict by threshold in Level-2: Precision {0:g}, Recall {1:g}, F1 {2:g}, AUPRC {3:g}"
                        .format(test_pre_second, test_rec_second, test_F1_second, test_prc_second))
            logger.info("Predict by threshold in Level-3: Precision {0:g}, Recall {1:g}, F1 {2:g}, AUPRC {3:g}"
                        .format(test_pre_third, test_rec_third, test_F1_third, test_prc_third))
            logger.info("Predict by threshold in Level-4: Precision {0:g}, Recall {1:g}, F1 {2:g}, AUPRC {3:g}"
                        .format(test_pre_fourth, test_rec_fourth, test_F1_fourth, test_prc_fourth))

            # Save the prediction result
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            dh.create_prediction_file(output_file=SAVE_DIR + "/predictions.json", data_id=test_data.patent_id,
                                      all_labels=true_labels, all_predict_labels=predicted_labels,
                                      all_predict_scores=predicted_scores)

    logger.info("All Done.")


if __name__ == '__main__':
    test_harnn()

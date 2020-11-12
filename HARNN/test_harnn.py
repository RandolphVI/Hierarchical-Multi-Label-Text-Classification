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


def create_input_data(data: dict):
    return zip(data['pad_seqs'], data['section'], data['subsection'], data['group'],
               data['subgroup'], data['onehot_labels'], data['labels'])


def test_harnn():
    """Test HARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels(args, args.test_file, word2idx)

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
            alpha = graph.get_operation_by_name("alpha").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            first_scores = graph.get_operation_by_name("first-output/scores").outputs[0]
            second_scores = graph.get_operation_by_name("second-output/scores").outputs[0]
            third_scores = graph.get_operation_by_name("third-output/scores").outputs[0]
            fourth_scores = graph.get_operation_by_name("fourth-output/scores").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/scores|second-output/scores|third-output/scores|fourth-output/scores|output/scores"

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(create_input_data(test_data)), args.batch_size, 1, shuffle=False)

            # Collect the predictions here
            true_labels = []
            predicted_labels = []
            predicted_scores = []

            # Collect for calculating metrics
            true_onehot_labels = [[], [], [], [], []]
            predicted_onehot_scores = [[], [], [], [], []]
            predicted_onehot_labels = [[], [], [], [], []]

            for batch_test in batches:
                x, sec, subsec, group, subgroup, y_onehot, y = zip(*batch_test)

                y_batch_test_list = [y_onehot, sec, subsec, group, subgroup]

                feed_dict = {
                    input_x: x,
                    input_y_first: sec,
                    input_y_second: subsec,
                    input_y_third: group,
                    input_y_fourth: subgroup,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    alpha: args.alpha,
                    is_training: False
                }
                batch_global_scores, batch_first_scores, batch_second_scores, batch_third_scores, batch_fourth_scores = \
                    sess.run([scores, first_scores, second_scores, third_scores, fourth_scores], feed_dict)

                batch_scores = [batch_global_scores, batch_first_scores, batch_second_scores,
                                batch_third_scores, batch_fourth_scores]

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores[0], threshold=args.threshold)

                # Add results to collection
                for labels in y:
                    true_labels.append(labels)
                for labels in batch_predicted_labels_ts:
                    predicted_labels.append(labels)
                for values in batch_predicted_scores_ts:
                    predicted_scores.append(values)

                for index in range(len(predicted_onehot_scores)):
                    for onehot_labels in y_batch_test_list[index]:
                        true_onehot_labels[index].append(onehot_labels)
                    for onehot_scores in batch_scores[index]:
                        predicted_onehot_scores[index].append(onehot_scores)
                    # Get one-hot prediction by threshold
                    predicted_onehot_labels_ts = \
                        dh.get_onehot_label_threshold(scores=batch_scores[index], threshold=args.threshold)
                    for onehot_labels in predicted_onehot_labels_ts:
                        predicted_onehot_labels[index].append(onehot_labels)

            # Calculate Precision & Recall & F1
            for index in range(len(predicted_onehot_scores)):
                test_pre = precision_score(y_true=np.array(true_onehot_labels[index]),
                                           y_pred=np.array(predicted_onehot_labels[index]), average='micro')
                test_rec = recall_score(y_true=np.array(true_onehot_labels[index]),
                                        y_pred=np.array(predicted_onehot_labels[index]), average='micro')
                test_F1 = f1_score(y_true=np.array(true_onehot_labels[index]),
                                   y_pred=np.array(predicted_onehot_labels[index]), average='micro')
                test_auc = roc_auc_score(y_true=np.array(true_onehot_labels[index]),
                                         y_score=np.array(predicted_onehot_scores[index]), average='micro')
                test_prc = average_precision_score(y_true=np.array(true_onehot_labels[index]),
                                                   y_score=np.array(predicted_onehot_scores[index]), average="micro")
                if index == 0:
                    logger.info("[Global] Predict by threshold: Precision {0:g}, Recall {1:g}, "
                                "F1 {2:g}, AUC {3:g}, AUPRC {4:g}"
                                .format(test_pre, test_rec, test_F1, test_auc, test_prc))
                else:
                    logger.info("[Local] Predict by threshold in Level-{0}: Precision {1:g}, Recall {2:g}, "
                                "F1 {3:g}, AUPRC {4:g}".format(index, test_pre, test_rec, test_F1, test_prc))

            # Save the prediction result
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            dh.create_prediction_file(output_file=SAVE_DIR + "/predictions.json", data_id=test_data['id'],
                                      true_labels=true_labels, predict_labels=predicted_labels,
                                      predict_scores=predicted_scores)
    logger.info("All Done.")


if __name__ == '__main__':
    test_harnn()

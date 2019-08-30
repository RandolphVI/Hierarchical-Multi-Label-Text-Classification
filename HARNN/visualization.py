# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import numpy as np
import tensorflow as tf

from utils import checkmate as cm
from utils import data_helpers as dh

# Parameters
# ==================================================

logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()))

MODEL = input("☛ Please input the model file you want to test, it should be like(1490175368): ")

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("✔︎ The format of your input is legal, now loading to next step...")

TRAININGSET_DIR = '../data/Train.json'
VALIDATIONSET_DIR = '../data/Validation.json'
TESTSET_DIR = '../data/Sample.json'
MODEL_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data")
tf.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.flags.DEFINE_integer("pad_seq_len", 150, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("beta", 1.0, "Weight of global losses in loss cal")
tf.flags.DEFINE_float("alpha", 0.75, "Weight of hierarchical violation in loss cal")
tf.flags.DEFINE_string("num_classes_list", "9,128,661", "Number of labels list (depends on the task)")
tf.flags.DEFINE_integer("total_classes", 798, "Total number of labels list (depends on the task)")
tf.flags.DEFINE_integer("top_num", 5, "Number of top K prediction classes (default: 5)")
tf.flags.DEFINE_float("threshold", 0.5, "Threshold for prediction classes (default: 0.5)")

# Test Parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 1)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def normalization(visual_list, visual_len):
    min_weight = min(visual_list[:visual_len])
    max_weight = max(visual_list[:visual_len])
    margin = max_weight - min_weight

    result = []
    for i in range(visual_len):
        value = (visual_list[i] - min_weight) / margin
        result.append(value)
    return result


def visualize():
    """visualize HARNN model."""

    # Load data
    logger.info("✔︎ Loading data...")
    logger.info("Recommended padding Sequence length is: {0}".format(FLAGS.pad_seq_len))

    logger.info("✔︎ Test data processing...")
    test_data = dh.load_data_and_labels(FLAGS.test_data_file, FLAGS.num_classes_list, FLAGS.total_classes,
                                        FLAGS.embedding_dim, data_aug_flag=False)

    logger.info("✔︎ Test data padding...")
    x_test, y_test, y_test_tuple = dh.pad_data(test_data, FLAGS.pad_seq_len)
    x_test_content, y_test_labels = test_data.abstract_content, test_data.labels

    # Load harnn model
    BEST_OR_LATEST = input("☛ Load Best or Latest Model?(B/L): ")

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("✘ The format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST.upper() == 'B':
        logger.info("✔︎ Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
    else:
        logger.info("✔︎ Loading latest model...")
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
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
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            beta = graph.get_operation_by_name("beta").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            first_visual = graph.get_operation_by_name("first-output/visual").outputs[0]
            second_visual = graph.get_operation_by_name("second-output/visual").outputs[0]
            third_visual = graph.get_operation_by_name("third-output/visual").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/visual|second-output/visual|third-output/visual|output/scores"

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(zip(x_test, y_test, y_test_tuple, x_test_content, y_test_labels)),
                                    FLAGS.batch_size, 1, shuffle=False)

            for batch_test in batches:
                x_batch_test, y_batch_test, y_batch_test_tuple, \
                x_batch_test_content, y_batch_test_labels = zip(*batch_test)

                y_batch_test_first = [i[0] for i in y_batch_test_tuple]
                y_batch_test_second = [j[1] for j in y_batch_test_tuple]
                y_batch_test_third = [k[2] for k in y_batch_test_tuple]

                feed_dict = {
                    input_x: x_batch_test,
                    input_y_first: y_batch_test_first,
                    input_y_second: y_batch_test_second,
                    input_y_third: y_batch_test_third,
                    input_y: y_batch_test,
                    dropout_keep_prob: 1.0,
                    beta: FLAGS.beta,
                    is_training: False
                }
                batch_first_visual, batch_second_visual, batch_third_visual, batch_scores = \
                    sess.run([first_visual, second_visual, third_visual, scores], feed_dict)

                seq_len = len(x_batch_test_content[0])
                pad_len = len(batch_first_visual[0])

                if seq_len >= pad_len:
                    length = pad_len
                else:
                    length = seq_len

                # print(seq_len, pad_len, length)
                final_first_visual = normalization(batch_first_visual[0].tolist(), length)
                final_second_visual = normalization(batch_second_visual[0].tolist(), length)
                final_third_visual = normalization(batch_third_visual[0].tolist(), length)

                visual_list = [final_first_visual, final_second_visual, final_third_visual]
                print(visual_list)

                f = open('attention.html', 'w')
                f.write('<html style="margin:0;padding:0;"><body style="margin:0;padding:0;">\n')
                f.write('<div style="margin:25px;">\n')
                for k in range(len(visual_list)):
                    f.write('<p style="margin:10px;">\n')
                    for i in range(seq_len):
                        alpha = "{:.2f}".format(visual_list[k][i])
                        word = x_batch_test_content[0][i]
                        f.write('\t<span style="margin-left:3px;background-color:rgba(255,0,0,{0})">{1}</span>\n'
                                .format(alpha, word))
                    f.write('</p>\n')
                f.write('</div>\n')
                f.write('</body></html>')
                f.close()

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    visualize()

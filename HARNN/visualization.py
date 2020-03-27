# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys
import time
import logging

sys.path.append('../')
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser

args = parser.parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("tflog", "logs/Test-{0}.log".format(time.asctime()))

CPT_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_CPT_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'output/' + MODEL


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
    """Visualize HARNN model."""

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels(args.test_file, args.num_classes_list, args.total_classes,
                                        args.word2vec_file, data_aug_flag=False)

    logger.info("Data padding...")
    x_test, y_test, y_test_tuple = dh.pad_data(test_data, args.pad_seq_len)
    x_test_content, y_test_labels = test_data.abstract_content, test_data.labels

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
            first_visual = graph.get_operation_by_name("first-output/visual").outputs[0]
            second_visual = graph.get_operation_by_name("second-output/visual").outputs[0]
            third_visual = graph.get_operation_by_name("third-output/visual").outputs[0]
            fourth_visual = graph.get_operation_by_name("fourth-output/visual").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "first-output/visual|second-output/visual|third-output/visual|fourth-output/visual|output/scores"

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-harnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(zip(x_test, y_test, y_test_tuple, x_test_content, y_test_labels)),
                                    args.batch_size, 1, shuffle=False)

            for batch_test in batches:
                x_batch_test, y_batch_test, y_batch_test_tuple, \
                x_batch_test_content, y_batch_test_labels = zip(*batch_test)

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
                batch_first_visual, batch_second_visual, batch_third_visual, batch_fourth_visual, batch_scores = \
                    sess.run([first_visual, second_visual, third_visual, fourth_visual, scores], feed_dict)

                seq_len = len(x_batch_test_content[0])
                pad_len = len(batch_first_visual[0])
                length = (pad_len if seq_len >= pad_len else seq_len)


                # print(seq_len, pad_len, length)
                final_first_visual = normalization(batch_first_visual[0].tolist(), length)
                final_second_visual = normalization(batch_second_visual[0].tolist(), length)
                final_third_visual = normalization(batch_third_visual[0].tolist(), length)
                final_fourth_visual = normalization(batch_fourth_visual[0].tolist(), length)

                visual_list = [final_first_visual, final_second_visual, final_third_visual, final_fourth_visual]
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

    logger.info("Done.")


if __name__ == '__main__':
    visualize()

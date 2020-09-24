# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import tensorflow as tf


class TextHARNN(object):
    """A HARNN for text classification."""

    def __init__(
            self, sequence_length, vocab_size, embedding_type, embedding_size, lstm_hidden_size, attention_unit_size,
            fc_hidden_size, num_classes_list, total_classes, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y_first = tf.placeholder(tf.float32, [None, num_classes_list[0]], name="input_y_first")
        self.input_y_second = tf.placeholder(tf.float32, [None, num_classes_list[1]], name="input_y_second")
        self.input_y_third = tf.placeholder(tf.float32, [None, num_classes_list[2]], name="input_y_third")
        self.input_y_fourth = tf.placeholder(tf.float32, [None, num_classes_list[3]], name="input_y_fourth")
        self.input_y = tf.placeholder(tf.float32, [None, total_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.alpha = tf.placeholder(tf.float32, name="alpha")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        def _attention(input_x, num_classes, name=""):
            """
            Attention Layer.

            Args:
                input_x: [batch_size, sequence_length, lstm_hidden_size * 2]
                num_classes: The number of i th level classes.
                name: Scope name.
            Returns:
                attention_weight: [batch_size, num_classes, sequence_length]
                attention_out: [batch_size, lstm_hidden_size * 2]
            """
            num_units = input_x.get_shape().as_list()[-1]
            with tf.name_scope(name + "attention"):
                W_s1 = tf.Variable(tf.truncated_normal(shape=[attention_unit_size, num_units],
                                                       stddev=0.1, dtype=tf.float32), name="W_s1")
                W_s2 = tf.Variable(tf.truncated_normal(shape=[num_classes, attention_unit_size],
                                                       stddev=0.1, dtype=tf.float32), name="W_s2")
                # attention_matrix: [batch_size, num_classes, sequence_length]
                attention_matrix = tf.map_fn(
                    fn=lambda x: tf.matmul(W_s2, x),
                    elems=tf.tanh(
                        tf.map_fn(
                            fn=lambda x: tf.matmul(W_s1, tf.transpose(x)),
                            elems=input_x,
                            dtype=tf.float32
                        )
                    )
                )
                attention_weight = tf.nn.softmax(attention_matrix, name="attention")
                attention_out = tf.matmul(attention_weight, input_x)
                attention_out = tf.reduce_mean(attention_out, axis=1)
            return attention_weight, attention_out

        def _fc_layer(input_x, name=""):
            """
            Fully Connected Layer.

            Args:
                input_x: [batch_size, *]
                name: Scope name.
            Returns:
                fc_out: [batch_size, fc_hidden_size]
            """
            with tf.name_scope(name + "fc"):
                num_units = input_x.get_shape().as_list()[-1]
                W = tf.Variable(tf.truncated_normal(shape=[num_units, fc_hidden_size],
                                                    stddev=0.1, dtype=tf.float32), name="W")
                b = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
                fc = tf.nn.xw_plus_b(input_x, W, b)
                fc_out = tf.nn.relu(fc)
            return fc_out

        def _local_layer(input_x, input_att_weight, num_classes, name=""):
            """
            Local Layer.

            Args:
                input_x: [batch_size, fc_hidden_size]
                input_att_weight: [batch_size, num_classes, sequence_length]
                num_classes: Number of classes.
                name: Scope name.
            Returns:
                logits: [batch_size, num_classes]
                scores: [batch_size, num_classes]
                visual: [batch_size, sequence_length]
            """
            with tf.name_scope(name + "output"):
                num_units = input_x.get_shape().as_list()[-1]
                W = tf.Variable(tf.truncated_normal(shape=[num_units, num_classes],
                                                    stddev=0.1, dtype=tf.float32), name="W")
                b = tf.Variable(tf.constant(value=0.1, shape=[num_classes], dtype=tf.float32), name="b")
                logits = tf.nn.xw_plus_b(input_x, W, b, name="logits")
                scores = tf.sigmoid(logits, name="scores")

                # shape of visual: [batch_size, sequence_length]
                visual = tf.multiply(input_att_weight, tf.expand_dims(scores, -1))
                visual = tf.nn.softmax(visual)
                visual = tf.reduce_mean(visual, axis=1, name="visual")
            return logits, scores, visual

        def _linear(input_, output_size, initializer=None, scope="SimpleLinear"):
            """
            Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k].

            Args:
                input_: a tensor or a list of 2D, batch x n, Tensors.
                output_size: int, second dimension of W[i].
                initializer: The initializer.
                scope: VariableScope for the created subgraph; defaults to "SimpleLinear".
            Returns:
                A 2D Tensor with shape [batch x output_size] equal to
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
            Raises:
                ValueError: if some of the arguments has unspecified or wrong shape.
            """

            shape = input_.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
            input_size = shape[1]

            # Now the computation.
            with tf.variable_scope(scope):
                W = tf.get_variable("W", [input_size, output_size], dtype=input_.dtype)
                b = tf.get_variable("b", [output_size], dtype=input_.dtype, initializer=initializer)

            return tf.nn.xw_plus_b(input_, W, b)

        def _highway_layer(input_, size, num_layers=1, bias=-2.0):
            """
            Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wx + b); h = relu(W'x + b')
            z = t * h + (1 - t) * x
            where t is transform gate, and (1 - t) is carry gate.
            """

            for idx in range(num_layers):
                h = tf.nn.relu(_linear(input_, size, scope=("highway_h_{0}".format(idx))))
                t = tf.sigmoid(_linear(input_, size, initializer=tf.constant_initializer(bias),
                                       scope=("highway_t_{0}".format(idx))))
                output = t * h + (1. - t) * input_
                input_ = output

            return output

        # Embedding Layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], minval=-1.0, maxval=1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="embedding")
            self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input_x)
            # Average Vectors
            # [batch_size, embedding_size]
            self.embedded_sentence_average = tf.reduce_mean(self.embedded_sentence, axis=1)

        # Bi-LSTM Layer
        with tf.name_scope("Bi-lstm"):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # forward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # backward direction cell
            if self.dropout_keep_prob is not None:
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

            # Creates a dynamic bidirectional recurrent neural network
            # shape of `outputs`: tuple -> (outputs_fw, outputs_bw)
            # shape of `outputs_fw`: [batch_size, sequence_length, lstm_hidden_size]

            # shape of `state`: tuple -> (outputs_state_fw, output_state_bw)
            # shape of `outputs_state_fw`: tuple -> (c, h) c: memory cell; h: hidden state
            outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                             self.embedded_sentence, dtype=tf.float32)
            # Concat output
            self.lstm_out = tf.concat(outputs, axis=2)  # [batch_size, sequence_length, lstm_hidden_size * 2]
            self.lstm_out_pool = tf.reduce_mean(self.lstm_out, axis=1)  # [batch_size, lstm_hidden_size * 2]

        # First Level
        self.first_att_weight, self.first_att_out = _attention(self.lstm_out, num_classes_list[0], name="first-")
        self.first_local_input = tf.concat([self.lstm_out_pool, self.first_att_out], axis=1)
        self.first_local_fc_out = _fc_layer(self.first_local_input, name="first-local-")
        self.first_logits, self.first_scores, self.first_visual = _local_layer(
            self.first_local_fc_out, self.first_att_weight, num_classes_list[0], name="first-")

        # Second Level
        self.second_att_input = tf.multiply(self.lstm_out, tf.expand_dims(self.first_visual, -1))
        self.second_att_weight, self.second_att_out = _attention(
            self.second_att_input, num_classes_list[1], name="second-")
        self.second_local_input = tf.concat([self.lstm_out_pool, self.second_att_out], axis=1)
        self.second_local_fc_out = _fc_layer(self.second_local_input, name="second-local-")
        self.second_logits, self.second_scores, self.second_visual = _local_layer(
            self.second_local_fc_out, self.second_att_weight, num_classes_list[1], name="second-")

        # Third Level
        self.third_att_input = tf.multiply(self.lstm_out, tf.expand_dims(self.second_visual, -1))
        self.third_att_weight, self.third_att_out = _attention(
            self.third_att_input, num_classes_list[2], name="third-")
        self.third_local_input = tf.concat([self.lstm_out_pool, self.third_att_out], axis=1)
        self.third_local_fc_out = _fc_layer(self.third_local_input, name="third-local-")
        self.third_logits, self.third_scores, self.third_visual = _local_layer(
            self.third_local_fc_out, self.third_att_weight, num_classes_list[2], name="third-")

        # Fourth Level
        self.fourth_att_input = tf.multiply(self.lstm_out, tf.expand_dims(self.third_visual, -1))
        self.fourth_att_weight, self.fourth_att_out = _attention(
            self.fourth_att_input, num_classes_list[3], name="fourth-")
        self.fourth_local_input = tf.concat([self.lstm_out_pool, self.fourth_att_out], axis=1)
        self.fourth_local_fc_out = _fc_layer(self.fourth_local_input, name="fourth-local-")
        self.fourth_logits, self.fourth_scores, self.fourth_visual = _local_layer(
            self.fourth_local_fc_out, self.fourth_att_weight, num_classes_list[3], name="fourth-")

        # Concat
        # shape of ham_out: [batch_size, fc_hidden_size * 4]
        self.ham_out = tf.concat([self.first_local_fc_out, self.second_local_fc_out,
                                  self.third_local_fc_out, self.fourth_local_fc_out], axis=1)

        # Fully Connected Layer
        self.fc_out = _fc_layer(self.ham_out)

        # Highway Layer
        with tf.name_scope("highway"):
            self.highway = _highway_layer(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Global scores
        with tf.name_scope("global-output"):
            num_units = self.h_drop.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[num_units, total_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[total_classes], dtype=tf.float32), name="b")
            self.global_logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.global_scores = tf.sigmoid(self.global_logits, name="scores")

        with tf.name_scope("output"):
            self.local_scores = tf.concat([self.first_scores, self.second_scores,
                                           self.third_scores, self.fourth_scores], axis=1)
            self.scores = tf.add(self.alpha * self.global_scores, (1 - self.alpha) * self.local_scores, name="scores")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            def cal_loss(labels, logits, name):
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
                losses = tf.reduce_mean(tf.reduce_sum(losses, axis=1), name=name + "losses")
                return losses

            # Local Loss
            losses_1 = cal_loss(labels=self.input_y_first, logits=self.first_logits, name="first_")
            losses_2 = cal_loss(labels=self.input_y_second, logits=self.second_logits, name="second_")
            losses_3 = cal_loss(labels=self.input_y_third, logits=self.third_logits, name="third_")
            losses_4 = cal_loss(labels=self.input_y_fourth, logits=self.fourth_logits, name="fourth_")
            local_losses = tf.add_n([losses_1, losses_2, losses_3, losses_4], name="local_losses")

            # Global Loss
            global_losses = cal_loss(labels=self.input_y, logits=self.global_logits, name="global_")

            # L2 Loss
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add_n([local_losses, global_losses, l2_losses], name="loss")
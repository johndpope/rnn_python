import tensorflow as tf
from Utilities import sequence_example_utilities
from Utilities import constants
from tensorflow.python.util.nest import flatten

def _build_rnn_cell(nr_layers, attention_length, input_keep_prob, output_keep_prob):

    cells = []

    # create a basic lstm cell
    for i in nr_layers:
        cell = tf.contrib.rnn.BasicLSTMCell(i)
        #attention wrapper for the first layer
        if attention_length and not cells:
            cell = tf.contrib.rnn.AttentionCellWrapper(
                cell, attention_length, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
        cells.append(cell)

    return tf.contrib.rnn.MultiRNNCell(cells)


#GENERATE TENSOR

def _init_generate(configuration, input_size):
    return tf.placeholder(tf.int64, [configuration.batch_size, None, input_size])

#ENCODER OF THE SEQ2SEQ MODEL
#
# def _init_encoder(configuration, inputs, length):
#     # create rnn cells
#     cell = build_rnn_cell(0, configuration.rnn_layer_size, configuration.attention_length, 0.5)
#
#     # no preinitial state for the moment
#     init_cell = cell.zero_state(configuration.batch_size, tf.int64)
#
#     # create rnn
#     return tf.nn.dynamic_rnn(cell, inputs, sequence_length= length, initial_state=init_cell)
#
#
# #DECODER OF THE SEQ2SEQ MODEL
#
# def _init_decoder(configuration, encoder_outputs, encoder_final_state):
#
#     cell = build_rnn_cell(1, configuration.rnn_layer_size, configuration.attention_length, 0.5)


def _flat_seq(seq, length):

    def flatten_unpadded_sequences():
        return tf.reshape(seq, [-1] + seq.shape.as_list()[2:])

    def flatten_padded_sequences():
        indices = tf.where(tf.sequence_mask(length))
        return tf.gather_nd(seq, indices)

    return tf.cond(tf.equal(tf.reduce_min(length),
                 tf.shape(seq)[1]),
                 flatten_unpadded_sequences,
                 flatten_padded_sequences)


def build_graph(filename, mode, configuration):

    if mode>=3:
        raise ValueError("Mode parameter is not correct.")

    #creates a default graph
    g = tf.Graph()

    #override the current default graph
    with g.as_default():

        inputs, labels, length = None

        #INIT PLACEHOLDER
        if mode == 0 or mode == 1:
            inputs, labels, length = sequence_example_utilities.build_input(filename, batches=configuration.batch_size, nr_classes=configuration.encoder_decoder.keys_number())
        elif mode == 2:
            inputs = _init_generate(configuration, configuration.encoder_decoder.get_input_size())


        #MAKE CELL
        cell = _build_rnn_cell(configuration.rnn_layer_size, configuration.attention_length, 0.8, 1.0)
        initial_state = cell.zero_state(configuration.batch_size, tf.float32)

        #MAKE RNN
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, initial_state=initial_state)

        #CREATE THE WEIGHTED MATRIX, RESULTING IN LOGITS
        #reduce a dimension added by the batched
        flatten_outputs = _flat_seq(outputs, length)
        flatten_logits = tf.contrib.layers.fully_connected(flatten_outputs, configuration.encoder_decoder.keys_number(), activation_fn = None)

        if mode == 0 or mode == 1:

            #COMPARE WITH LOGITS USING SOFTMAX
            flatten_labels = _flat_seq(labels, length)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flatten_labels, logits=flatten_logits)

            #CORRECT LOGITS
            flatten_prediction = tf.argmax(flatten_logits, axis=1)
            correct_predictions = tf.to_float(tf.equal(flatten_labels, flatten_prediction))
            event_positions = tf.to_float(tf.not_equal(flatten_labels, constants.NO_CHORD_EVENT))
            no_event_positions = tf.to_float(tf.equal(flatten_labels, constants.NO_CHORD_EVENT))

            if mode == 0:

                #compute rnn parameters
                loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                perplexity = tf.reduce_mean(tf.exp(cross_entropy))
                accuracy = tf.reduce_mean(correct_predictions)
                event_accuracy = (
                    tf.reduce_sum(correct_predictions * event_positions) /
                    tf.reduce_sum(event_positions))
                no_event_accuracy = (
                    tf.reduce_sum(correct_predictions * no_event_positions) /
                    tf.reduce_sum(no_event_positions))

                optimizer = tf.train.AdagradOptimizer(learning_rate=configuration.learning_rate)

                train = tf.contrib.slim.learning.create_train_op(loss, optimizer)
                tf.add_to_collection('train_node', train)

                train_param = {
                    'loss': loss,
                    'perplexity': perplexity,
                    'accuracy': accuracy,
                    'event_accuracy': event_accuracy,
                    'no_event_accuracy': no_event_accuracy,
                }

                for k in train_param:
                    tf.summary.scalar(k, train_param[k])
                    tf.add_to_collection(k, train_param[k])

            elif mode == 1:

                #USE TF.SLIM FOR EVALUATING AFTER TRAIN AND SHOW RESULTS IN TENSOBOARD
                eval_param, update_ops = tf.contrib.metrics.aggregate_metric_map(
                    {
                        'loss': tf.metrics.mean(cross_entropy),
                        'metrics/accuracy': tf.metrics.accuracy(flatten_labels, flatten_prediction),
                        'metrics/per_class_accuracy':tf.metrics.mean_per_class_accuracy(
                                flatten_labels, flatten_prediction, configuration.encoder_decoder.keys_number),
                        'metrics/event_accuracy': tf.metrics.recall(event_positions, correct_predictions),
                        'metrics/no_event_accuracy': tf.metrics.recall(no_event_positions, correct_predictions),
                        'metrics/perplexity': tf.metrics.mean(tf.exp(cross_entropy)),
                    })
                for updates_op in update_ops.values():
                    tf.add_to_collection('eval_node', updates_op)

                for k in eval_param:
                    #see them in tensorboard
                    tf.summary.scalar(k, eval_param[k])
                    tf.add_to_collection(k, eval_param[k])

            elif mode == 2:

                res = tf.placeholder(tf.int64, [])
                flatten_softmax = tf.nn.softmax(
                    tf.div(flatten_logits, tf.fill([configuration.encoder_decoder.keys_number], res)))
                softmax = tf.reshape(flatten_softmax, [configuration.batch_size, -1, configuration.encoder_decoder.keys_number])

                tf.add_to_collection('inputs', inputs)
                tf.add_to_collection('res', res)
                tf.add_to_collection('softmax', softmax)

                for state in flatten(initial_state):
                    tf.add_to_collection('initial_state', state)
                for state in flatten(final_state):
                    tf.add_to_collection('final_state', state)

        return g
































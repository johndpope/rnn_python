import tensorflow as tf
from RNN import prepare_encoder_entry
import seq2seq as seq2seq


def _build_rnn_cell(mode, nr_layers, attention_length, dropout_keep_probability, size_layer_units=1):
    # create a basic lstm cell
    cell = tf.contrib.rnn.BasicLSTMCell(size_layer_units)

    # decoder cell
    if mode == 1:
        # add attention wraper to the cell
        cell = tf.contrib.rnn.AttentionCellWrapper(cell, attention_length, state_is_tuple=True)

        # add dropout wrapper to the cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_probability)

        return tf.contrib.rnn.MultiRNNCell([cell] * range(nr_layers))

    # encoder cell
    return tf.contrib.rnn.MultiRNNCell([cell] * range(nr_layers))


#GENERATE TENSOR

def _init_generate(configuration, input_size):
    return tf.placeholder(tf.int64, [configuration.batch_size, None, input_size])


#BATCHES FROM INPUT FILES

def _init_train_eval(mode, configuration):

    data_preparator = prepare_encoder_entry.MusicEntryOneHot(configuration.encoder_decoder)

    #train
    if mode == 0:
        database = data_preparator.collect_data("train_database")

    else:
        database = data_preparator.collect_data("eval_database")
    return data_preparator.prepare_input(database, configuration.batch_size)



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



def build_graph(file_path, mode, configuration):

    if mode>=3:
        raise ValueError("Mode parameter is not correct.")

    input_size = configuration.encoder_decoder.get_input_size()

    #creates a default graph
    g = tf.Graph()

    #override the current default graph
    with g.as_default():

        #INIT PLACEHOLDER

        if mode == 0 or mode == 1:
            inputs, keys = _init_train_eval(mode, configuration)
        elif mode == 2:
            inputs = _init_generate(configuration, input_size)

        cell = _build_rnn_cell(mode,configuration,)
























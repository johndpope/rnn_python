import tensorflow as tf
from Utilities import utilities

#define multilayer rnn cell
def build_rnn_cell(nr_layers, attention_length, dropout_keep_probability, size_layer_units = 1):

    #create a basic lstm cell
    cell = tf.contrib.rnn.BasicLSTMCell(size_layer_units)

    #add attention wraper to the cell
    cell = tf.contrib.rnn.AttentionCellWrapper(cell, attention_length, state_is_tuple=True)

    #add dropout wrapper to the cell
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_probability)

    return tf.tf.contrib.rnn.MultiRNNCell([cell] * range(nr_layers))


def build_graph(file_path, mode, configuration):

    if mode>=3:
        raise ValueError("Mode parameter is not correct.")

    input_size = configuration.encoder_decoder.get_input_size()

    #creates a default graph
    g = tf.Graph()

    #override the current default graph
    with g.as_default():

        #if in training mode or in evaluation mode, get batched from input files
        if mode == 0 or mode == 1:
            inputs, keys = utilities.read_from_file(file_path, configuration.batch_size, input_size)
        #if in generate mode, create an input tensor for the song
        else:
            inputs = tf.placeholder(tf.int64, [configuration.batch_size, None, input_size])

        # create rnn cells
        cell = build_rnn_cell(configuration.rnn_layer_size, configuration.attention_length, 0.5)

        # no preinitial state for the moment
        init_cell = cell.zero_state(configuration.batch_size, tf.int64)

        #create rnn
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_cell)











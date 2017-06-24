import tensorflow as tf
from Utilities import constants

#This function makes a SequenceExample. A SequenceExample is a data format in which data for training can be stored.
#It contains a key-value (features); each feature contains a key (sring) and a value (BytesList, FloatList, Int64List)
#and a context (features which apply to the entire context).

def build_sequence_example_chords(inputs, keys):

    input_feature = [tf.train.Feature(int64_list = tf.train.Int64List(value=inp)) for inp in inputs]
    key_feature = [tf.train.Feature(int64_list = tf.train.Int64List(value=[key])) for key in keys]

    feature_list = tf.train.FeatureLists(feature_list =
    {
        'keys': tf.train.FeatureList(feature=key_feature),
        'inputs': tf.train.FeatureList(feature=input_feature)
    })

    return tf.train.SequenceExample(feature_lists = feature_list)


def seq_ex_writer(filename, seq_ex):

    writer = tf.python_io.TFRecordWriter(filename)
    writer.write(seq_ex.SerializeToString())
    writer.close()

def build_input(filename, batches, nr_classes):

    file_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(file_queue)

    seq_features = {
        'inputs': tf.FixedLenSequenceFeature(shape = [nr_classes], dtype=tf.int64),
        'keys': tf.FixedLenSequenceFeature(shape = [], dtype=tf.int64)
    }

    _, sequence = tf.parse_single_sequence_example(serialized_example, sequence_features=seq_features)

    length = tf.shape(sequence['inputs'])[0]

    return tf.train.batch(tensors = [sequence['inputs'], sequence['keys'], length], enqueue_many=True, batch_size=batches, dynamic_pad=True)




import tensorflow as tf
from Utilities import constants

#This function makes a SequenceExample. A SequenceExample is a data format in which data for training can be stored.
#It contains a key-value (features); each feature contains a key (sring) and a value (BytesList, FloatList, Int64List)
#and a context (features which apply to the entire context).

def build_sequence_example_chords(inputs, keys):

    input_feature = [tf.train.Feature(int64_list = tf.train.BytesList(value=inp)) for inp in inputs]
    key_feature = [tf.train.Feature(int64_list = tf.train.BytesList(value=key)) for key in keys]

    feature_list = tf.train.FeatureLists(feature_list =
    {
        'target': tf.train.FeatureList(feature=key_feature),
        'source': tf.train.FeatureList(feature=input_feature)
    })

    return tf.train.SequenceExample(feature_lists = feature_list)


def write_to_file(file, seq_example):

    writer = tf.python_io.TFRecordWriter(file)
    writer.write(seq_example.SerializeToString())
    writer.close()



def build_input(seq_example, batches):

    serialized_features = seq_example.SerializeToString()

    seq_features = {
        'inputs': tf.FixedLenSequenceFeature(shape = [constants.MAX_PITCH - constants.MIN_PITCH + 2], dtype=tf.int64),
        'keys': tf.FixedLenSequenceFeature(shape = [constants.MAX_SIMULTAN_NOTES], dtype=tf.int64)
    }

    _, sequence = tf.parse_single_sequence_example(serialized_features, sequence_features=seq_features)

    return tf.train.batch(tensors = [sequence['inputs'], sequence['keys']], enqueue_many=True, batch_size=batches, dynamic_pad=True)




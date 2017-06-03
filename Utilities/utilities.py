import tensorflow as tf

'''This function makes a SequenceExample. A SequenceExample is a data format in which data for training can be stored.
      It contains a key-value (features); each feature contains a key (sring) and a value (BytesList, FloatList, Int64List)
      and a context (features which apply to the entire context).
   '''

def build_sequence_example(inputs, keys):
    input_feature = [tf.train.Feature(int64_list = tf.train.Int64List(value=inp)) for inp in inputs]
    key_feature = [tf.train.Feature(int64_list = tf.train.Int64List(value=[key])) for key in keys]

    feature_list = tf.train.FeatureLists(feature_list =
    {
        'keys': tf.train.FeatureList(feature=key_feature),
        'inputs': tf.train.FeatureList(feature=input_feature)
    })

    return tf.train.SequenceExample(feature_lists = feature_list)


#------------------------------------------------------------
#from tensorflow library - adapted for this rnn

def read_from_file(files, batch_size, input_size):

    file_queue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    sequence_features = {
        'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                             dtype=tf.int64),
        'keys': tf.FixedLenSequenceFeature(shape=[],
                                             dtype=tf.int64)}

    _, sequence = tf.parse_single_sequence_example(serialized_example, sequence_features=sequence_features)

    length = tf.shape(sequence['inputs'])[0]

    return tf.train.batch(
        [sequence['inputs'], sequence['labels'], length],
        batch_size=batch_size,
        capacity=500,
        num_threads=4,
        dynamic_pad=True,
        allow_smaller_final_batch=False)



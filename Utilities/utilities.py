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
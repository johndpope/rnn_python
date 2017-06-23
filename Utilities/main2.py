import tensorflow as tf
import seq2seq as seq2seq

k = tf.ones([2,3,4], dtype=tf.int32)
j = tf.transpose(k, [1,0,2])






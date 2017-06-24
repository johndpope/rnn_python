import tensorflow as tf
import seq2seq as seq2seq
import music21 as music21


k = tf.ones([2,3,4], dtype=tf.int32)
j = tf.transpose(k, [1,0,2])


chord = music21.chord.Chord("C C")
print(chord.pitchedCommonName)






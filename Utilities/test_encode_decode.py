from RNN import music_info, encode_decode_one_chord, encode_decode_chord_sequence
from Utilities import music_utilities
from Utilities import sequence_example_utilities
import tensorflow as tf


music_info = music_info.MusicInfo("/home/bristina/PycharmProjects/seq2seq/proto/proto_database")
music_info.write_file_info()
#
music_inf = music_utilities.read_database_info()
print(music_inf[0])
print(music_inf[1])
#
encoder_decoder = encode_decode_chord_sequence.EncodeDecodeOneHotSeqChords(encode_decode_one_chord.EncodeDecodeOneChord(music_info=music_inf))
music_utilities.prepare_chords_input(music_info.midi_database, encoder_decoder)

batch = sequence_example_utilities.build_input("TRAIN", 5, music_inf[0])


sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(2):
    final_res = sess.run(batch)
    print(final_res)
    print("NEW")
coord.request_stop()
coord.join(threads)

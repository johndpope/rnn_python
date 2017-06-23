from proto import midi_pb2
from RNN import prepare_encoder_entry
from RNN import encode_decode_chord
from RNN import encode_decode_one_note
from Utilities import constants
from operator import add
import tensorflow as tf
import seq2seq as seq2seq
# #
# # import tensorflow as tf
# #
# # g = tf.Graph()
# # with g.as_default():
# #     input_value = tf.constant(1.0, name = "input")
# #     weight = tf.Variable(6.9, name = "weight")
# #     output_value = tf.add(input_value, weight, name="output")
# #     assert input_value.graph is g
# #     assert weight.graph, output_value.graph is g
# #
# #
#
# # tf.global_variables_initializer()
# #
# # summary_writer = tf.summary.FileWriter("./try", g)
# #
# #
# #
midi_database = midi_pb2.MidiDatabase()
#
# #Read the existing files
f = open("/home/bristina/PycharmProjects/seq2seq/proto/proto_database", "rb")
midi_database.ParseFromString(f.read())
f.close()

# dict = {}
#
# # #
# music_entry = prepare_encoder_entry.MusicEntryOneHot(encode_decode_chord.EncodeDecodeOneHotEncodingChords(encode_decode_one_note.EncodeDecodeOneNote(44,92)))
# i = 0
#
# for song in midi_database.midi_song:
#     notes = music_entry.prepare_notes(song)
#     for j in range(len(notes[0])):
#         new_chord = music_entry.encoder_decoder.notes_to_input(notes, j)
#         if new_chord in dict.values():
#             pass
#         else:
#             dict[i] = new_chord
#             i += 1
#
# i = 1




# #
# # # # # for i in midi_database.midi_song[0].track[1].notes:
# # # # #     print(i.start_time_s)
# # # # #print(midi_database.midi_song[0].file_name)
# # # # #
# j = music_entry.make_file_tfrecord([midi_database.midi_song[0], midi_database.midi_song[1], midi_database.midi_song[2]])
#
# train_input_pipeline = seq2seq.data.input_pipeline.TFRecordInputPipeline({  "files": ["TRAIN"],
#         "source_field": "source",
#         "target_field": "target",
#         "source_delimiter": " ",
#         "target_delimiter": " ",}, tf.contrib.learn.ModeKeys.TRAIN)
#
# data_prov = train_input_pipeline.make_data_provider()
# features_and_labels = train_input_pipeline.read_from_data_provider(data_prov)
#
# a = 0
#
# sess = tf.Session()
#
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# for i in range(2):
#     final_image = sess.run(features_and_labels)
#     print(final_image)
#     print("NEW")
# coord.request_stop()
# coord.join(threads)


#
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
# for i in range(0,2):
#     k = sess.run(j[0])
#     print(k)
#
# coord.request_stop()
# coord.join(threads)


# # li = [0]*6
# l = [1]
# li[0]=1
# # li[0] += l
# print(li)

#
# l, r = tf.unstack(tf.shape([[1,2,3], [1,2,5], [1,2,3]]))
# l = tf.shape([[1,2], [1,2]])
# #print(l)
# #print(r)
#
#
# image_batch = tf.train.batch(tensors = [j[0]['inputs']], enqueue_many=True, batch_size=4, dynamic_pad=True)
#
# sess = tf.Session()
#
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# for i in range(2):
#     final_image = sess.run(image_batch)
#     print(final_image)
#     print("NEW")
# coord.request_stop()
# coord.join(threads)
# #
#
#



#

#


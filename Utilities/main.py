from proto import midi_pb2
from RNN import prepare_data
from RNN import encode_decode_sequence
from RNN import encode_decode_one_note

# import tensorflow as tf
#
# g = tf.Graph()
# with g.as_default():
#     input_value = tf.constant(1.0, name = "input")
#     weight = tf.Variable(6.9, name = "weight")
#     output_value = tf.add(input_value, weight, name="output")
#     assert input_value.graph is g
#     assert weight.graph, output_value.graph is g
#
#
# sess = tf.Session()
# tf.global_variables_initializer()
#
# summary_writer = tf.summary.FileWriter("./try", g)



midi_database = midi_pb2.MidiDatabase()

#Read the existing files
f = open("/home/bristina/PycharmProjects/seq2seq/proto/proto_database", "rb")
midi_database.ParseFromString(f.read())
f.close()

music_entry = prepare_data.MusicEntryOneHot(encode_decode_sequence.EncodeDecodeOneHotEncoding(encode_decode_one_note.EncodeDecodeOneNote))

# for i in midi_database.midi_song[0].track[1].notes:
#     print(i.start_time_s)
#print(midi_database.midi_song[0].file_name)

for j in music_entry.prepare_notes(midi_database.midi_song[0]):
    print(j)

# li = [0]*6
# l = [1]
# li[0]=1
# # li[0] += l
# print(li)

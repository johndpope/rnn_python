from RNN import beam_search_from_tf, encode_decode_one_chord, encode_decode_chord_sequence
from Utilities import constants, music_utilities
import tensorflow as tf


def main():

    #get the trained graph
    graph = tf.train.import_meta_graph(meta_graph_or_file='home/bristina/music_graph/graph')
    sess = tf.Session(graph=graph)

    #initiate the encoder
    music_info = music_utilities.read_database_info()
    encoder_decoder = encode_decode_chord_sequence.EncodeDecodeOneHotSeqChords(encode_decode_one_chord.EncodeDecodeOneChord(music_info=music_info))

    #use C major - 1/4 to initialize the sequence
    c_major = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
    c_major[6] = 1
    c_major[10] = 1
    c_major[13] = 1

    end = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
    end[1] += 3

    events_not_encoded = [c_major, end]
    events = []

    for i in range(len(events_not_encoded)):
        events.append(encoder_decoder.notes_to_input(events_not_encoded, i))


    #params
    branch_factor = 1
    steps_per_it = 1
    temperature = 1

    #choose number of chords
    number_of_chords = 100
    number_of_steps = number_of_chords * constants.MIN_DURATION

    for i in range(len(constants.MIDI_TO_GENERATE)):
        chord_sequence = beam_search_from_tf.generate_song(sess,
                                                 encoder_decoder,
                                                 events,
                                                 number_of_steps,
                                                 temperature,
                                                 int(music_info[0]/3),
                                                 branch_factor,
                                                 steps_per_it)
        music_utilities.generate_from_sequence(chord_sequence)






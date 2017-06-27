from abc import ABCMeta, abstractmethod
from Utilities import sequence_example_utilities
import numpy as np
from RNN import encode_decode_one_chord
from Utilities import constants



class IEncodeDecodeSequenceChord:

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_input_size(self): raise NotImplementedError

    #returns number of keys
    @abstractmethod
    def keys_number(self): raise NotImplementedError

    #returns the input vector for a note
    @abstractmethod
    def notes_to_input(self, notes, position): raise NotImplementedError

    # returns a key for a note
    @abstractmethod
    def notes_to_key(self, notes, position): raise NotImplementedError

    # returns a note for a specified key
    @abstractmethod
    def key_to_note(self, key): raise NotImplementedError

    #notes_sequence: list of notes
    def encode(self, chords_sequence):

        inputs = []
        labels = []

        for i in range(len(chords_sequence)-1):
            inputs.append(self.notes_to_input(chords_sequence, i))
            labels.append(self.notes_to_key(chords_sequence, i+1))
        return sequence_example_utilities.build_sequence_example_chords(inputs, labels)

class EncodeDecodeOneHotSeqChords(IEncodeDecodeSequenceChord):

    def __init__(self, one_chord_encode_decode):
        if not issubclass(type(one_chord_encode_decode), encode_decode_one_chord.IEncodeDecodeOneChord):
            raise TypeError("The argument given is not an instance of a encoder_decoder_one_chord class")
        self.one_chord_encode_decode = one_chord_encode_decode


    def get_input_size(self):
        return self.one_chord_encode_decode.get_class_size()


    def keys_number(self):
        return self.one_chord_encode_decode.get_class_size()

    #one hot encoding - returns a vector with keys number size which is all 0, except the position of the class to which the note belongs
    def notes_to_input(self, chords, position):

        one_hot_vector = [0] * self.get_input_size()

        last_key = self.notes_to_key(chords, position)
        one_hot_vector[last_key] += 1
        return one_hot_vector

    def notes_to_key(self, chords, position):

        chord = chords[position]
        return self.one_chord_encode_decode.encode_one_chord(chord)

    def key_to_note(self, key):
        return self.one_chord_encode_decode.decode_one_chord(key)
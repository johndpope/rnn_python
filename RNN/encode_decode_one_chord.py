from Utilities import sequence_example_utilities, constants
from abc import ABCMeta, abstractmethod
from functools import reduce

class IEncodeDecodeOneChord:

    __metaclass__ = ABCMeta

    @abstractmethod
    #return an integer value representing the number of symbols used
    def get_class_size(self): raise NotImplementedError

    @abstractmethod
    #return a representation for a note
    def encode_one_chord(self, chord): raise NotImplementedError

    @abstractmethod
    #return a note from a representation
    def decode_one_chord(self, chords): raise NotImplementedError


class EncodeDecodeOneChord(IEncodeDecodeOneChord):

    def __init__(self, music_info):
       self.music_info = music_info

    def get_class_size(self):
        return len(self.music_info.class_dictionary)

    #makes the chord as a base10 number from base 4
    def encode_one_chord(self, chord):

        chords_number = [chord[i]*pow(constants.MAX_SIMULTAN_NOTES+1,i) for i in range(0, len(chord))]
        unseq_key = reduce(lambda x, y: x+y, chords_number)

        ordered_keys = sorted(self.music_info.class_dictionary.keys())
        for i in range(0, len(ordered_keys)):
            if ordered_keys[i] == unseq_key:
                return i

    def decode_one_chord(self, index):

        ordered_keys = sorted(self.music_info.class_dictionary.keys())
        unseq_key = ordered_keys[index]
        return self.music_info.class_dictionary[unseq_key]
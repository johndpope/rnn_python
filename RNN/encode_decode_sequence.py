from abc import ABCMeta, abstractmethod
from Utilities import utilities

class IEncodeDecodeSequence:

    __metaclass__ = ABCMeta

    @abstractmethod
    def input_size(self): raise NotImplementedError

    #returns number of keys
    @abstractmethod
    def keys_number(self): raise NotImplementedError

    #returns the input vector for the note at a specified position
    @abstractmethod
    def notes_to_input(self, notes, position): raise NotImplementedError

    # returns a key for the note at a specified position
    @abstractmethod
    def notes_to_key(self, notes, position): raise NotImplementedError

    # returns a note for a specified key
    @abstractmethod
    def key_to_note(self, key, notes): raise NotImplementedError

    #notes_sequence: list of notes
    def encode(self, notes_sequence):

        inputs, keys = zip(
            *[(self.notes_to_input(notes_sequence, i), self.notes_to_key(notes_sequence, i)) for i in
              range(len(notes_sequence) - 1)])
        return utilities.build_sequence_example(inputs, keys)





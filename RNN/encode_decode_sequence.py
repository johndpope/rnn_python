from abc import ABCMeta, abstractmethod
from Utilities import utilities
import numpy as np
from RNN import encode_decode_one_note

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
    def key_to_note(self, key): raise NotImplementedError

    #notes_sequence: list of notes
    #returns a SequenceExample object
    def encode(self, notes_sequence):

        inputs, keys = zip(
            *[(self.notes_to_input(notes_sequence, i), self.notes_to_key(notes_sequence, i)) for i in
              range(len(notes_sequence) - 1)])
        return utilities.build_sequence_example(inputs, keys)

    #returns a batch containing the last parameter if is_first = false and all the sequence if is_first is true
    def get_batches(self, notes_sequence, is_first = False):

        final_input = []

        for notes in notes_sequence:
            current_input = []
            if is_first == True:
                current_input =  [self.notes_to_input(notes_sequence, i) for i in range(len(notes) - 1)]
            else:
                current_input = self.notes_to_input(notes_sequence, len(notes_sequence) - 1)
            final_input.append(current_input)

        return final_input

    #return a vector which evaluates the probability of each element to be the desider sequence - the smaller, the better
    def apply_log_likelihood(self, notes_sequences, softmax):

        final_likelihood = []

        if len(notes_sequences) != len(softmax):
            raise ValueError("Softmax list and note sequences list should have the same length")

        for i in range(len(notes_sequences)):
            if len(softmax[i]) >= len(notes_sequences[i]):
                raise ValueError("The softmax vector corresponding to a sequence cannot have a length grater or equal with the sequence length.")
            #compute number of notes
            notes_evaluate = len(notes_sequences[i]) - len(softmax[i])
            log_likelihood = 0.0
            softmax_position = 0
            for position in range(len(notes_sequences[i]), notes_evaluate):
                i = self.notes_to_key(notes_sequences[i], position)
                log_likelihood += np.log(softmax[i][softmax_position][i])
                softmax_position += 1

            final_likelihood.append(log_likelihood)

        return final_likelihood


class EncodeDecodeOneHotEncoding(IEncodeDecodeSequence):

    def __init__(self,one_note_encode_decode):
        if not isinstance(one_note_encode_decode, encode_decode_one_note.IEncodeDecodeOneNote):
            return TypeError ("The argument given is not an instance of a encoder_decoder_one_note class")
        self.one_note_encode_decode = one_note_encode_decode

    def input_size(self):
        return self.one_note_encode_decode.symbols_numbers

    def keys_number(self):
        return self.one_note_encode_decode.symbols_numbers

    #one hot encoding - returns a vector with keys number size which is all 0, except the position of the class to which the note belongs
    def notes_to_input(self, notes, position):
        final_input = [0]*self.input_size()
        final_input[self.one_note_encode_decode.encode_one_note(notes[position])] = 1
        return final_input

    def notes_to_key(self, notes, position):
        return self.one_note_encode_decode.encode_one_note(notes, position)

    def key_to_note(self, key):
        return self.one_note_encode_decode.decode_one_note(key)






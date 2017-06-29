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
    def key_to_note(self, key, position): raise NotImplementedError

    #notes_sequence: list of notes
    def encode(self, chords_sequence):

        inputs = []
        labels = []

        for i in range(len(chords_sequence)-1):
            inputs.append(self.notes_to_input(chords_sequence, i))
            labels.append(self.notes_to_key(chords_sequence, i+1))
        return sequence_example_utilities.build_sequence_example_chords(inputs, labels)

    def get_inputs_batch(self, event_sequences):

        inputs_batch = []

        for events in event_sequences:
            inputs = [self.notes_to_input(events,i) for i in range(len(events))]
            inputs_batch.append(inputs)

        return inputs_batch


    def extend_event_sequences(self, event_sequences, softmax):

        num_classes = len(softmax[0][0])
        chosen_classes = []
        for i in range(len(event_sequences)):
            chosen_class = np.random.choice(num_classes, p=softmax[i][-1])
            event = self.key_to_note(chosen_class, event_sequences[i])
            event_sequences[i].append(event)
            chosen_classes.append(chosen_class)

        return chosen_classes

    #return a vector which evaluates the probability of each element to be the desider sequence - the smaller, the better
    def evaluate_log_likelihood(self, notes_sequences, softmax):

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

    def key_to_note(self, key, position):
        return self.one_chord_encode_decode.decode_one_chord(key)
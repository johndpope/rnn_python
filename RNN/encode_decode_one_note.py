from Utilities import utilities, constants
from abc import ABCMeta, abstractmethod


class IEncodeDecodeOneNote:

    __metaclass__ = ABCMeta

    @abstractmethod
    #return an integer value representing the number of symbols used
    def symbols_number(self): raise NotImplementedError

    @abstractmethod
    #return a representation for a note
    def encode_one_note(self): raise NotImplementedError

    @abstractmethod
    #return a note from a representation
    def decode_one_note(self): raise NotImplementedError


class EncodeDecodeOneNote(IEncodeDecodeOneNote):

    """Encodes melodies as following(labels):
        0 - pause
        [1, value] - note_on, value = pitch
        [2, value] - note_off, value = pitch
    """

    def __init__(self, min_note, max_note, note_duration):

       if(min_note < constants.MIN_PITCH):
            raise ValueError("Minimum pitch is too low")
       if(max_note > constants.MIN_PITCH):
            raise ValueError("Maximum pitch is too low")
       if(max_note < min_note):
           raise ValueError("Highest pitch must be higher than lowest pitch")

       self.min_note = min_note
       self.max_note = max_note
       self.note_duration = note_duration


    @property
    def symbols_number(self):
        return (self.max_note - self.min_note)*2 +1


    def encode_one_note(self, note):

        event, pitch = note
        if pitch > utilities.min_note:
            return (event, pitch-constants.min_note)

        return 0

    def decode_one_note(self, index):

        event, pitch = index
        if pitch > utilities.min_note:
            return (event, pitch+constants.min_note)

        return 0

















from Utilities import sequence_example_utilities, constants
from abc import ABCMeta, abstractmethod


class IEncodeDecodeOneNote:

    __metaclass__ = ABCMeta

    @abstractmethod
    #return an integer value representing the number of symbols used
    def get_class_size(self): raise NotImplementedError

    @abstractmethod
    #return a representation for a note
    def encode_one_note(self, note): raise NotImplementedError

    @abstractmethod
    #return a note from a representation
    def decode_one_note(self, index): raise NotImplementedError


class EncodeDecodeOneNote(IEncodeDecodeOneNote):

    """Encodes melodies as following(labels):
        0 - pause
        1 - note_off
        [2, value] - note_on, value = pitch
    """

    def __init__(self, min_note, max_note):

       if(min_note < constants.MIN_PITCH):
            raise ValueError("Minimum pitch is too low")
       if(max_note > constants.MAX_PITCH):
            raise ValueError("Maximum pitch is too low")
       if(max_note < min_note):
           raise ValueError("Highest pitch must be higher than lowest pitch")

       self.min_note = min_note
       self.max_note = max_note

    def get_class_size(self):
        return (self.max_note - self.min_note) + 2

    def encode_one_note(self, note):

        #note event
        if note >= self.min_note:
            return note-self.min_note+2

        #note off event or pause event
        return note+2

    def decode_one_note(self, index):

        #note index
        if index>=2:
            return (index-2)+self.min_note

        #note off event or pause event
        return index - 2

















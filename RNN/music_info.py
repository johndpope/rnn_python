from proto import midi_pb2
from abc import ABCMeta, abstractmethod
from Utilities import constants, music_utilities

from functools import reduce


class MusicInfo:

    __metaclass__ = ABCMeta

    def __init__(self, database_name):
        self.midi_database = self.collect_data(database_name)
        self.class_dictionary = self.get_dictionary()

    #get all the data from proto file
    def collect_data(self, database_name):

        midi_database = midi_pb2.MidiDatabase()

        # Read the existing files
        f = open(database_name, "rb")
        midi_database.ParseFromString(f.read())
        f.close()

        return midi_database

    def get_dictionary(self):

        no_event = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
        no_event[0] += 2

        class_dict = {'3' : no_event[0]}

        def encode_one_note(note):

            if note >= constants.MIN_PITCH:
                return note - constants.MIN_PITCH + 2

            return note + 2

        for song in self.midi_database.midi_song:
            notes = music_utilities.prepare_notes(song)
            for j in range(0, len(notes[0])):
                new_chord = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
                for i in range(0, constants.MAX_SIMULTAN_NOTES):
                    new_chord[encode_one_note(notes[i][j])] += 1

                chords_number = [new_chord[i] * pow(constants.MAX_SIMULTAN_NOTES + 1, i) for i in range(0, len(new_chord))]
                unseq_key = reduce(lambda x, y: x + y, chords_number)

                if unseq_key not in class_dict.keys():
                    class_dict[unseq_key] = new_chord

        return class_dict


    def write_file(self):
        



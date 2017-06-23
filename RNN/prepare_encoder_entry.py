from proto import midi_pb2
from abc import ABCMeta, abstractmethod
from Utilities import constants, utilities
import tensorflow as tf


class IMusicEntry:

    __metaclass__ = ABCMeta

    def __init__(self, encoder_decoder):
        self.encoder_decoder = encoder_decoder

    #get all the data from proto file
    def collect_data(self, database_name):
        midi_database = midi_pb2.MidiDatabase()

        # Read the existing files
        f = open(database_name, "rb")
        midi_database.ParseFromString(f.read())
        f.close()

        return midi_database

    @abstractmethod
    def prepare_notes(self, song): raise NotImplementedError

    #midi_list : list of melodies
    #returns a sequence example, according to the chosen encoder
    def prepare_input(self, midi_list): raise NotImplementedError

class MusicEntryOneHot(IMusicEntry):


    def prepare_input(self, midi_list, batch_size):

        data = []

        for i in range(0, constants.MAX_SIMULTAN_NOTES):
            data.append([])

        for song in midi_list.midi_song:
            note_list = self.prepare_notes(song)
            x = len(note_list[0]) % batch_size
            if x != 0:
                concat = [-2] * x
                for i in range(0, len(note_list)):
                    note_list[i] += concat
            for i in range(0, len(note_list)):
                data[i] += note_list[i]

        encoded_data = self.encoder_decoder.encode(data)
        batched_data = utilities.build_input(encoded_data, batch_size)
        return batched_data


    def _get_classes(self, midi_database):

        dict = {}

        number_of_classes = 0

        for song in midi_database.midi_song:
            notes = self.prepare_notes(song)
            for j in range(len(notes[0])):
                new_chord = self.encoder_decoder.notes_to_input(notes, j)
                if new_chord not in dict.values():
                    dict[number_of_classes] = new_chord
                    number_of_classes += 1

        return dict



    #returns a list with the given note/event with min duration
    def make_list_min_duration(self, note, min_duration):

        output_list = []

        if note > 0:
            output_list += ([note] * (min_duration - 1))
            output_list += ([constants.NOTE_OFF_EVENT])
            return output_list

        output_list += ([constants.PAUSE_EVENT] * min_duration)
        return output_list


    #only notes from the second track
    def prepare_notes(self, song):

        output_list = []

        for i in range(constants.MAX_SIMULTAN_NOTES):
            output_list.append([])

        tempos = song.track[0].tempo
        second_track_notes = song.track[1].notes

        current_tempo = 0
        last_note = [0.0] * constants.MAX_SIMULTAN_NOTES
        i = 0

        while i < len(second_track_notes):

            channel_number = 0
            current_note = second_track_notes[i]
            i += 1

            #compute current tempo - if there is a new tempo for the current note, replace the last tempo with the new one
            if current_tempo+1 < len(tempos) and tempos[current_tempo + 1].time > current_note.start_time_s:
                current_tempo += 1

            #pause time from the last note until current_note
            pause_time = (current_note.start_time_s - last_note[channel_number])/(tempos[current_tempo].tempo_value/1000000.0)
            f_pause_time = pause_time / constants.MIN_DURATION

            #pause between notes on all channels
            blanks_n = self.make_list_min_duration(constants.PAUSE_EVENT, int(round(f_pause_time)))
            output_list[channel_number] += blanks_n

            last_note[channel_number] = current_note.end_time_s


            # how many quavers have the current note
            note_duration_min = current_note.duration / constants.MIN_DURATION

            # actual note in quavers
            actual_note = self.make_list_min_duration(current_note.pitch, int(round(note_duration_min)))
            output_list[channel_number] += actual_note

            while i < len(second_track_notes) and second_track_notes[i].start_time_s == current_note.start_time_s:

                if channel_number < constants.MAX_SIMULTAN_NOTES-1:
                    current_note = second_track_notes[i]

                    channel_number += 1
                    i += 1

                    # pause time from the last note until current_note
                    pause_time = (current_note.start_time_s - last_note[channel_number]) / (tempos[current_tempo].tempo_value / 1000000.0)

                    f_pause_time = pause_time / constants.MIN_DURATION

                    # pause between notes on all channels
                    blanks_n = self.make_list_min_duration(constants.PAUSE_EVENT, int(round(f_pause_time)))
                    output_list[channel_number] += blanks_n



                    # how many quavers have the current note
                    note_duration_min = current_note.duration / constants.MIN_DURATION

                    # actual note in quavers
                    actual_note = self.make_list_min_duration(current_note.pitch, int(round(note_duration_min)))
                    output_list[channel_number] += actual_note

                    last_note[channel_number] = current_note.end_time_s

                else:
                    i+=1


        total_length = len(output_list[0])
        for i in range(1, constants.MAX_SIMULTAN_NOTES):
            if len(output_list[i]) < total_length:
                l = [constants.PAUSE_EVENT] * (total_length - len(output_list[i]))
                output_list[i] += l


        return output_list















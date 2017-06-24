from Utilities import constants, sequence_example_utilities
import tensorflow as tf
import music21 as music21


dict_chords = {
    44: 'G#',
    45: 'A',
    46: 'A#',
    47: 'B',
    48: 'C',
    49: 'C#',
    50: 'D',
    51: 'D#',
    52: 'E',
    53: 'F',
    54: 'F#',
    55: 'G',
    56: 'G#',
    57: 'A',
    58: 'A#',
    59: 'B',
    60: 'C',
    61: 'C#',
    62: 'D',
    63: 'D#',
    64: 'E',
    65: 'F',
    66: 'F#',
    67: 'G',
    68: 'G#',
    69: 'A',
    70: 'A#',
    71: 'B',
    72: 'C',
    73: 'C#',
    74: 'D',
    75: 'D#',
    76: 'E',
    77: 'F',
    78: 'F#',
    79: 'G',
    80: 'G#',
    81: 'A',
    82: 'A#',
    83: 'B',
    84: 'C',
    85: 'C#',
    86: 'D',
    87: 'D#',
    88: 'E',
    89: 'F',
    90: 'F#',
    91: 'G'

}


# returns a list with the given note/event with min duration
def make_list_min_duration(note, min_duration):
    output_list = []

    if note > 0:
        output_list += ([note] * (min_duration - 1))
        output_list += ([constants.NOTE_OFF_EVENT])
        return output_list

    output_list += ([constants.PAUSE_EVENT] * min_duration)
    return output_list


def normalize_note(note):
    while note.pitch < constants.MIN_PITCH:
        note.pitch += 12
    while note.pitch > constants.MAX_PITCH:
        note.pitch -= 12

    return note.pitch


# only notes from the second track
def prepare_notes(song):

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
        new_pitch = normalize_note(current_note)
        current_note.pitch = new_pitch

        i += 1

        # compute current tempo - if there is a new tempo for the current note, replace the last tempo with the new one
        if current_tempo + 1 < len(tempos) and tempos[current_tempo + 1].time > current_note.start_time_s:
            current_tempo += 1

        # pause time from the last note until current_note
        pause_time = (current_note.start_time_s - last_note[channel_number]) / (tempos[current_tempo].tempo_value / 1000000.0)
        f_pause_time = pause_time / constants.MIN_DURATION

        # pause between notes on all channels
        blanks_n = make_list_min_duration(constants.PAUSE_EVENT, int(round(f_pause_time)))
        output_list[channel_number] += blanks_n

        last_note[channel_number] = current_note.end_time_s

        # how many quavers have the current note
        note_duration_min = current_note.duration / constants.MIN_DURATION

        # actual note in quavers
        actual_note = make_list_min_duration(current_note.pitch, int(round(note_duration_min)))
        output_list[channel_number] += actual_note

        while i < len(second_track_notes) and second_track_notes[i].start_time_s == current_note.start_time_s:

            if channel_number < constants.MAX_SIMULTAN_NOTES - 1:
                current_note = second_track_notes[i]
                new_pitch = normalize_note(current_note)
                current_note.pitch = new_pitch

                channel_number += 1
                i += 1

                # pause time from the last note until current_note
                pause_time = (current_note.start_time_s - last_note[channel_number]) / (
                tempos[current_tempo].tempo_value / 1000000.0)

                f_pause_time = pause_time / constants.MIN_DURATION

                # pause between notes on all channels
                blanks_n = make_list_min_duration(constants.PAUSE_EVENT, int(round(f_pause_time)))
                output_list[channel_number] += blanks_n

                # how many quavers have the current note
                note_duration_min = current_note.duration / constants.MIN_DURATION

                # actual note in quavers
                actual_note = make_list_min_duration(current_note.pitch, int(round(note_duration_min)))
                output_list[channel_number] += actual_note

                last_note[channel_number] = current_note.end_time_s

            else:
                i += 1

    partial_length = max(len(output_list[0]), len(output_list[1]))
    total_length = max(partial_length, len(output_list[2]))
    for i in range(0, constants.MAX_SIMULTAN_NOTES):
        if len(output_list[i]) < total_length:
            l = [constants.PAUSE_EVENT] * (total_length - len(output_list[i]))
            output_list[i] += l

    return output_list


def get_chords(notes):

    chords = []
    for j in range(0, len(notes[0])):
        new_chord = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
        for i in range(0, constants.MAX_SIMULTAN_NOTES):
            note = notes[i][j]
            if note>=constants.MIN_PITCH:
                note = note - constants.MIN_PITCH + 2
            else:
                note += 2
            new_chord[note] += 1
        chords.append(new_chord)

    return chords


def prepare_chords_input(midi_database, encoder_decoder):

    song_nr = 0

    writer = tf.python_io.TFRecordWriter('TRAIN')

    for song in midi_database:

        note_list = prepare_notes(song)
        chords = get_chords(note_list)
        for chord in chords:
            chordd = ""
            for i in range(len(chord)):
                if chord[i] != 0 and i > 1:
                    note = i + constants.MIN_PITCH - 2
                    chordd = chordd + dict_chords[note] + " "
            if chordd != "":
                chord_music = music21.chord.Chord(chordd)
                print(chord_music.pitchedCommonName)
        # seq_ex = encoder_decoder.encode(chords)
        # writer.write(seq_ex.SerializeToString())
        # print(song_nr)
        # song_nr += 1

    writer.close()
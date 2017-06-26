from RNN import encode_decode_one_chord, encode_decode_chord_sequence
from Utilities import constants
from RNN import music_info

class ConfigRnn():

    #In this class all the hyperparameters will be set

    #default configuration
    def __init__(self):
        self.info_file = "info"
        self.encode_decoder = encode_decode_chord_sequence.EncodeDecodeOneHotSeqChords(encode_decode_one_chord.EncodeDecodeOneChord(music_info.MusicInfo(self.info_file)))
        self.learning_rate = 0.1
        self.batch_size = 12
        self.rnn_layer_size = 1
        self.nr_epochs_before_start = 10
        self.nr_training_steps = 1000
        self.truncated_backprop_length = (1.0/constants.MIN_DURATION)*5


    def set_info_file(self, file):
        self.info_file = file

    #   value: 1 - OneHotEncoding
    def set_encoder_decoder(self, value):

        try:
            if not isinstance(value, int):
                raise TypeError
        except TypeError:
                print("Error: Encoder-Decoder must be an integer.Set default encoder")
                self.encode_decoder = encode_decode_one_chord.EncodeDecodeOneChord(encode_decode_one_chord.EncodeDecodeOneChord(self.info_file))
                return self

        if value == 0:
            self.encode_decoder = encode_decode_chord_sequence.EncodeDecodeOneHotSeqChords(encode_decode_one_chord.EncodeDecodeOneChord(self.info_file))
            return self

        try:
            raise ValueError
        except ValueError:
                print("There is no encoder with this number.Set default encoder")
                self.encode_decoder = encode_decode_chord_sequence.EncodeDecodeOneHotSeqChords(encode_decode_one_chord.EncodeDecodeOneChord(self.info_file))
                return self


    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def set_rnn_layer_size(self, rnn_layer_size):
        self.rnn_layer_size = rnn_layer_size
        return self

    def set_number_training_steps(self, nr_training_steps):
        self.nr_training_steps = nr_training_steps
        return self

    def set_early_stop_after(self, nr_epochs):
        self.nr_epochs_before_start = nr_epochs
        return self

    def set_attention_length(self, attention_length):
        self.attention_length = attention_length
        return self

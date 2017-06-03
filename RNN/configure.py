from RNN import encode_decode_one_note, encode_decode_sequence


class ConfigRnn():

    #In this class all the hyperparameters will be set

    #default configuration
    def __init__(self):
        self.encode_decoder = encode_decode_sequence.EncodeDecodeOneHotEncoding(encode_decode_one_note.EncodeDecodeOneNote)
        self.learning_rate = 0.0
        self.batch_size = 1
        self.rnn_layer_size = 1
        self.nr_epochs_before_start = 10
        self.nr_training_steps = 1000

    #   value: 1 - OneHotEncoding
    def set_encoder_decoder(self, value):

        try:
            if not isinstance(value, int):
                raise TypeError
        except TypeError:
                print("Error: Encoder-Decoder must be an integer.Set default encoder")
                self.encode_decoder = encode_decode_sequence.EncodeDecodeOneHotEncoding(encode_decode_one_note.EncodeDecodeOneNote)
                return self

        if value == 0:
            self.encode_decoder = encode_decode_sequence.EncodeDecodeOneHotEncoding(encode_decode_one_note.EncodeDecodeOneNote)
            return self

        try:
            raise ValueError
        except ValueError:
                print("There is no encoder with this number.Set default encoder")
                self.encode_decoder = encode_decode_sequence.EncodeDecodeOneHotEncoding(encode_decode_one_note.EncodeDecodeOneNote)
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

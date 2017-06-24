from RNN import configure_rnn
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from seq2seq.training import hooks
from seq2seq.training import utils as training_utils
from tensorflow import gfile
from RNN import music_info_scratch

def main(mode):

    #configure the rnn

    config = configure_rnn.ConfigRnn()
    config.set_batch_size(64) \
          .set_encoder_decoder(1) \
          .set_early_stop_after(10) \
          .set_learning_rate(10) \
          .set_number_training_steps(2000) \
          .set_rnn_layer_size([64, 64])








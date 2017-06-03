from RNN import configure

def main(mode):

    #configure the rnn

    config = configure.ConfigRnn()
    config.set_batch_size(64) \
          .set_encoder_decoder(1) \
          .set_early_stop_after(10) \
          .set_learning_rate(10) \
          .set_number_training_steps(2000) \
          .set_rnn_layer_size([64, 64])

    





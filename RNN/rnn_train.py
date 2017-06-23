from RNN import configure_rnn
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from seq2seq.training import hooks
from seq2seq.training import utils as training_utils
from tensorflow import gfile
from RNN import prepare_encoder_entry

def main(mode):

    #configure the rnn

    config = configure_rnn.ConfigRnn()
    config.set_batch_size(64) \
          .set_encoder_decoder(1) \
          .set_early_stop_after(10) \
          .set_learning_rate(10) \
          .set_number_training_steps(2000) \
          .set_rnn_layer_size([64, 64])



def make_train_seq2seq(output_dir, configure_run, configure_rnn):

    #output_dir - output directory for checkpoints

    config = run_config.RunConfig(
                tf_random_seed= configure_run.random_seed,
                save_checkpoints_secs= configure_run.checkpoint_secs,
                save_checkpoints_steps= configure_run.checkpoint_steps,
                keep_checkpoint_max= configure_run.checkpoint_max,
                keep_checkpoint_every_n_hours= configure_run.keep_checkpoint_every_n_hours,
                gpu_memory_fraction=configure_run.gpu_memory_fraction)

    config.tf_config.gpu_options.allow_growth = configure_run.gpu_allow_growth
    config.tf_config.log_device_placement = configure_run.log_device_placement

    if config.is_chief():
        gfile.MakeDirs(output_dir)

    prepare_inputs = prepare_encoder_entry.MusicEntryOneHot(configure_rnn.encode_decoder)
    dataset_train = prepare_inputs.collect_data("TRAIN")
    batches_train_fn = prepare_inputs.prepare_input(dataset_train, configure_rnn.batch_size)

    dataset_eval = prepare_inputs.collect_data("EVAL")
    batches_eval_fn = prepare_inputs.prepare_notes(dataset_eval, configure_rnn.batch_size)

    








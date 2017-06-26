from RNN import configure_rnn, configure_run
from RNN import rnn_graph
import tensorflow as tf
from Utilities import constants

def _run_train(graph, config, logdir):

    with graph.as_default():

        global_step = tf.train.get_or_create_global_step()

        # get graphs params
        loss = tf.get_collection('loss')[0]
        perplexity = tf.get_collection('perplexity')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_node')[0]

        values_dictionary = \
            {
                'Global Step': global_step,
                'Loss': loss,
                'Perplexity': perplexity,
                'Accuracy': accuracy
            }

        hooks_list = [tf.train.NanTensorHook(loss_tensor=loss),
                      tf.train.LoggingTensorHook(tensors = values_dictionary, every_n_iter=config.save_checkpoint_steps),
                      tf.train.StepCounterHook(output_dir=logdir, every_n_steps=config.save_checkpoint_steps)]

        #scaffold object for saving checkpoints info
        saver = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=config.keep_checkpoint_max))

        tf.contrib.training.train(
            train_op=train_op,
            logdir=logdir,
            scaffold=saver,
            hooks=hooks_list,
            save_checkpoint_steps = config.save_checkpoint_steps,
            save_summaries_steps=config.save_summaries_steps)

def _run_eval(graph, config, logdir, eval_steps):

    with graph.as_default():
        global_step = tf.train.get_or_create_global_step()

        # get graphs params
        loss = tf.get_collection('loss')[0]
        perplexity = tf.get_collection('metrics/perplexity')[0]
        accuracy = tf.get_collection('metrics/accuracy')[0]
        eval_op = tf.get_collection('eval_node')


        values_dictionary = \
            {
                'Global Step': global_step,
                'Loss': loss,
                'Perplexity': perplexity,
                'Accuracy': accuracy
            }

        hooks_list = [tf.train.LoggingTensorHook(tensors = values_dictionary, every_n_iter=config.save_checkpoint_steps)]

        tf.contrib.training.evaluate_repeteadly(
            checkpoint_dir = logdir,
            eval_ops = eval_op,
            feed_dict = values_dictionary,
            max_number_of_evaluation = eval_steps,
            hooks=hooks_list,
            eval_interval_secs=60,
            timeout= 300
        )


def main():

    tf.logging.set_verbosity(verbosity='INFO')

    #set all dirs for infos
    run_dir = constants.INFO_DIR + "run_info"
    train_dir = constants.INFO_DIR + "train_info"
    eval_dir = constants.INFO_DIR + "eval_info"

    #configure the rnn
    config_rnn = configure_rnn.ConfigRnn()
    config_rnn.set_batch_size(64) \
        .set_encoder_decoder(1) \
        .set_early_stop_after(10) \
        .set_learning_rate(10) \
        .set_number_training_steps(2000) \
        .set_rnn_layer_size([64, 64])

    #configure run
    config_run = configure_run.ConfigRun()

    mode = 0

    if mode == 0:
        graph = rnn_graph.build_graph("proto_train", mode, config_rnn)
        tf.logging.info("Start training.")
        _run_train(graph, config_run, train_dir)
        tf.logging.info("End training.")
    elif mode == 1:
        graph = rnn_graph.build_graph("proto_eval", mode, config_rnn)
        tf.logging.info("Start evaluation.")
        number_of_steps = constants.NUMBER_OF_FILES / configure_rnn.batch_size
        _run_eval(graph, config_run, eval_dir, number_of_steps)
        tf.logging.info("End evaluation.")






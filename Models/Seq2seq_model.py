import tensorflow as tf
import collections
from seq2seq import graph_utils
from seq2seq import losses as seq2seq_losses
from seq2seq.contrib.seq2seq.decoder import _transpose_batch_time
from seq2seq.graph_utils import templatemethod
from seq2seq.decoders.beam_search_decoder import BeamSearchDecoder
from seq2seq.inference import beam_search
from seq2seq.models.model_base import ModelBase, _flatten_dict


#MODIFIED CLASS FROM LIBRARY SEQ2SEQ
class Seq2SeqModel(ModelBase):

    def __init__(self, params, mode, name):
        super(Seq2SeqModel, self).__init__(params, mode, name)

    @staticmethod
    def default_params():
        params = ModelBase.default_params()
        params.update({
            "source.max_seq_len": 50,
            "source.reverse": True,
            "target.max_seq_len": 50,
            "inference.beam_search.beam_width": 0,
            "inference.beam_search.length_penalty_weight": 0.0,
            "inference.beam_search.choose_successors_fn": "choose_top_k",
            "optimizer.clip_embed_gradients": 0.1,
        })


        return params


    def _create_predictions(self, decoder_output, features, labels, losses=None):
        """Creates the dictionary of predictions that is returned by the model.
        """
        predictions = {}

        # Add features and, if available, labels to predictions
        predictions.update(_flatten_dict({"features": features}))
        if labels is not None:
            predictions.update(_flatten_dict({"labels": labels}))

        if losses is not None:
            predictions["losses"] = _transpose_batch_time(losses)

        # Decoders returns output in time-major form [T, B, ...]
        # Here we transpose everything back to batch-major for the user
        output_dict = collections.OrderedDict(
            zip(decoder_output._fields, decoder_output))
        decoder_output_flat = _flatten_dict(output_dict)
        decoder_output_flat = {
            k: _transpose_batch_time(v)
            for k, v in decoder_output_flat.items()
            }
        predictions.update(decoder_output_flat)

       #TODO PREDICT BACK WHAT COMES FROM DECODER

        return predictions
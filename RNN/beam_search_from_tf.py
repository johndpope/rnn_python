#THIS FILE IS A COPY OF BEAM SEARCH FROM MAGENTA

import tensorflow as tf
import numpy as np
from tensorflow.python.util.nest import map_structure
import copy
import heapq

def unbatch(batched_states, batch_size=1):
    final_output = []

    for i in range(batch_size):
        final_output.append(extract_state(batched_states, i))

    return final_output


def extract_state(batched_states, i):
    return map_structure(lambda x: x[i], batched_states)


def generate_step_for_batch(event_sequences, session, encoder_decoder, inputs, initial_state, temperature):

    assert len(event_sequences) == session.graph.get_collection('inputs')[0].shape[0].value

    graph_inputs = session.graph.get_collection('inputs')[0]
    graph_initial_state = session.graph.get_collection('initial_state')
    graph_final_state = session.graph.get_collection('final_state')
    graph_softmax = session.graph.get_collection('softmax')[0]
    graph_temperature = session.graph.get_collection('temperature')

    feed_dict = {graph_inputs: inputs,
                 tuple(graph_initial_state): initial_state}

    if graph_temperature:
      feed_dict[graph_temperature[0]] = temperature
    final_state, softmax = session.run([graph_final_state, graph_softmax], feed_dict)

    if softmax.shape[1] > 1:
      loglik = encoder_decoder.evaluate_log_likelihood(event_sequences, softmax[:, :-1, :])
    else:
      loglik = np.zeros(len(event_sequences))

    indices = encoder_decoder.extend_event_sequences(event_sequences, softmax)
    p = softmax[range(len(event_sequences)), -1, indices]

    return final_state, loglik + np.log(p)


def batch(states, batch_size=None):

  if batch_size and len(states) > batch_size:
    raise ValueError('Combined state is larger than the requested batch size')

  def stack_and_pad(*states):
      stacked = np.stack(states)
      if batch_size:
          stacked.resize([batch_size] + list(stacked.shape)[1:])
      return stacked

  return map_structure(stack_and_pad, *states)

def generate_step(event_sequences, session, encoder_decoder, inputs, initial_states, temperature):

    batch_size = session.graph.get_collection('inputs')[0].shape[0].value
    num_seqs = len(event_sequences)
    num_batches = int(np.ceil(num_seqs / float(batch_size)))

    final_states = []
    loglik = np.empty(num_seqs)

    # Add padding to fill the final batch.
    pad_amt = -len(event_sequences) % batch_size
    padded_event_sequences = event_sequences + [
        copy.deepcopy(event_sequences[-1]) for _ in range(pad_amt)]
    padded_inputs = inputs + [inputs[-1]] * pad_amt
    padded_initial_states = initial_states + [initial_states[-1]] * pad_amt

    for b in range(num_batches):
      i, j = b * batch_size, (b + 1) * batch_size
      pad_amt = max(0, j - num_seqs)
      batch_final_state, batch_loglik = generate_step_for_batch(
          padded_event_sequences[i:j],
          session,
          encoder_decoder,
          padded_inputs[i:j],
          batch(padded_initial_states[i:j], batch_size), temperature)
      final_states += unbatch(batch_final_state, batch_size)[:j - i - pad_amt]
      loglik[i:j - pad_amt] = batch_loglik[:j - i - pad_amt]

    return final_states, loglik

def generate_options(event_sequences, session, encoder_decoder, loglik, branch_factor, num_steps, inputs, initial_states, temperature):

    all_event_sequences = [copy.deepcopy(events) for events in event_sequences * branch_factor]
    all_inputs = inputs * branch_factor
    all_final_state = initial_states * branch_factor
    all_loglik = np.tile(loglik, (branch_factor,))

    for _ in range(num_steps):
      all_final_state, all_step_loglik = generate_step(all_event_sequences,
                                                       session,
                                                       encoder_decoder,
                                                       all_inputs,
                                                       all_final_state,
                                                       temperature)
      all_loglik += all_step_loglik

    return all_event_sequences, all_final_state, all_loglik

def prune_branches(event_sequences, final_states, loglik, k):

    indices = heapq.nlargest(k, range(len(event_sequences)),
                             key=lambda i: loglik[i])

    event_sequences = [event_sequences[i] for i in indices]
    final_states = [final_states[i] for i in indices]
    loglik = loglik[indices]

    return event_sequences, final_states, loglik

# beam_size = k => keep track of most likely k outputs
# number_of_steps => how many iterations to perform
# seq_dimension => the dimension of the sequence on witch the next outputs depend on

def beam_search(encoder_decoder, session, events_sequence, number_of_steps, temperature, beam_size, branch_factor, steps_per_it):

    graph_initial_state = session.graph.get_collection('initial_state')
    log_likelihood = tf.zeros(beam_size)

    current_states = unbatch(session.run(graph_initial_state))[0]
    relevant_states = [current_states] * beam_size

    first_it_steps = (number_of_steps - 1) % steps_per_it + 1
    event_sequences, final_state, loglik = generate_options(events_sequence,
                                                            session,
                                                            encoder_decoder,
                                                            log_likelihood,
                                                            branch_factor,
                                                            first_it_steps,
                                                            relevant_states,
                                                            inputs,
                                                            temperature)

    num_iterations = (number_of_steps - first_it_steps) / steps_per_it

    for _ in range(num_iterations):
        event_sequences, final_state, loglik = prune_branches(event_sequences,
                                                              final_state,
                                                              loglik,
                                                              k=beam_size)

        inputs = encoder_decoder.get_inputs_batch(event_sequences)

        event_sequences, final_state, loglik = generate_options(event_sequences,
                                                                session,
                                                                encoder_decoder,
                                                                loglik,
                                                                branch_factor,
                                                                steps_per_it,
                                                                inputs,
                                                                final_state,
                                                                temperature)

    # Prune to a single sequence.
    event_sequences, final_state, loglik = prune_branches(event_sequences,
                                                          final_state,
                                                          loglik,
                                                          k=1)

    return event_sequences[0]


def generate_song(session, encoder_decoder, events, beam_size, temperature, branch_factor, number_of_steps, steps_per_iteration):

    return beam_search(encoder_decoder, session, number_of_steps - len(events), temperature,
                                   beam_size, branch_factor, steps_per_iteration)




# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Benchmark for a beam search implementation.

This code requires TF 2.0 or newer.
Use `pip install tf-nightly-2.0-preview` to install.
Tip: Run the `pip install` command in a separate virtual environment
(Virtualenv, Anaconda) to avoid clobbering an existing TF installation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import benchmark_base

import numpy as np
import tensorflow as tf


# Implicitly assumed by code: OOV_TOKEN = 0
INIT_TOKEN = 1
EOS_TOKEN = 2

BEAM_SIZE = 5
NEG_INF = -1e10

MAX_SEQ_LEN = 50


class FakeDecoder(tf.keras.Model):
  """Stub decoder implementation that just returns a random array of logits."""

  def __init__(self, vocab_size=1000, hidden_size=256):
    super(FakeDecoder, self). __init__()
    self.vocab_size = vocab_size
    self.lstm = tf.keras.layers.LSTM(hidden_size, return_state=True)

  def get_initial_state(self, batch_size):
    return self.lstm.cell.get_initial_state(
        batch_size=batch_size, dtype=tf.float32)

  def call(self, x, hidden_state):
    np_logits = -np.random.random([x.shape[0], self.vocab_size])
    np_logits[:, EOS_TOKEN] = -10000  # don't allow beam search to exit early
    return tf.constant(np_logits, dtype=tf.float32), hidden_state


def get_best_alive(cumulative_logprob, eos_token, beam_size, vocab_size):
  """Get top_k sequences/log probs, masking out completed sequences.

  Args:
    cumulative_logprob: Tensor of shape [beam_size, vocab_size] with cumulative
        log-probability for each possible continuation of each beam.
    eos_token: ID of the end-of-sequence token.
    beam_size: beam size.
    vocab_size: vocab size.

  Returns:
    tuple of (chosen_sequences, alive_logprobs, alive_indices)
    chosen_sequences: Tensor of shape [beam_size] with the indices of the
        sequences that are being continued.
    alive_logprobs: Tensor of shape [beam_size] with log probs of the top K
        alive sequence continuations
    alive_indices: Tensor of shape [beam_size] with the token that comes next
        for each sequence continuation.
  """
  # Mask finished sequences with -INF so that top_k ignores them.
  cumulative_logprob = tf.where(
      tf.equal(tf.one_hot([eos_token] * beam_size, vocab_size), 1),
      tf.tile([[NEG_INF]], [beam_size, vocab_size]),
      cumulative_logprob)

  # [beam_size * vocab_size]
  flat_logprobs = tf.reshape(cumulative_logprob, [-1])
  alive_logprobs, alive_kn_indices = tf.nn.top_k(flat_logprobs, k=beam_size)
  # [beam_size], [beam_size]
  alive_indices, chosen_sequences = alive_kn_indices % vocab_size, alive_kn_indices // vocab_size
  return chosen_sequences, alive_logprobs, alive_indices


def get_best_sequences(beam_size, seq1, logprobs1, seq2, logprobs2):
  """Helper function to get top_k from two sets of sequences/log probs.

  Args:
    beam_size: Beam size.
    seq1: Tensor of shape [beam_size, max_seq_len]
    logprobs1: Tensors of shape [beam_size]
    seq2: Tensor of shape [beam_size, max_seq_len]
    logprobs2: Tensors of shape [beam_size]

  Returns:
    top beam_size sequences and log probs from the two sets as tensors of
    shape [beam_size, max_seq_len] and [beam_size]
  """
  both_seq = tf.concat([seq1, seq2], axis=0)
  both_logprobs = tf.concat([logprobs1, logprobs2], axis=0)
  chosen_logprobs, chosen_idx = tf.nn.top_k(both_logprobs, k=beam_size)
  return tf.gather(both_seq, chosen_idx), chosen_logprobs


def beam_search(decoder, init_hidden_state, init_token, eos_token, beam_size,
                vocab_size, max_seq_len=MAX_SEQ_LEN):
  """Beam search.

  Keeps beam_size living sequences at each iteration, and beam_size completed
  sequences at each iteration. Completes when all living sequences have dropped
  far enough in probability that no living sequences have any chance of beating
  one of the known completed sequences, or if the search limit has been reached.

  If, at the end, an incomplete sequence with max_seq_len has higher probability
  than any complete sequence, then it will be ranked higher than the completed
  sequence.

  Args:
    decoder: Decoder module.
    init_hidden_state: A hidden state representing decoding context. Should have
        a batch dimension with size 1.
    init_token: Token to seed decoding with.
    eos_token: Token to compare against to see if sequence is ended.
    beam_size: beam size.
    vocab_size: vocab size.
    max_seq_len: Maximum seq len before stopping and returning what we have.

  Returns:
    Tuple of sequences, log probs.
    sequences: Tensor of shape [beam_size, max_seq_len]
    log_probs: Tensor of shape [beam_size]
  """
  init_logits, hidden_state = decoder(tf.constant([init_token]),
                                      init_hidden_state)
  start_logprobs = tf.nn.log_softmax(tf.squeeze(init_logits))

  # Seed the starting sequences by executing decoder once and taking top k.
  # [beam_size], [beam_size]
  alive_logprobs, alive_indices = tf.nn.top_k(start_logprobs, k=beam_size)
  # [beam_size, max_seq_len]
  alive_sequences = tf.concat([
      tf.expand_dims(alive_indices, 1),
      tf.zeros([beam_size, max_seq_len - 1], dtype=tf.int32)], axis=1)
  # [[beam_size, hidden_size], ...]
  alive_hidden = tf.nest.map_structure(
      lambda s: tf.tile(s, [beam_size, 1]),
      hidden_state)

  # Seed finished sequences as the empty sequence, i.e. [<EOS>, 0, 0...] and
  # zeros everywhere else.
  # Mark all other sequences with logprob = -INF
  finished_sequences = eos_token * tf.one_hot(
      [0], beam_size * max_seq_len, dtype=tf.int32)
  finished_sequences = tf.reshape(finished_sequences, [beam_size, max_seq_len])
  finished_logprobs = tf.where(
      tf.equal(tf.one_hot(0, beam_size), 1),
      tf.tile([start_logprobs[eos_token]], [beam_size]),
      tf.tile([NEG_INF], [beam_size]))

  for i in tf.range(1, max_seq_len):
    # [beam_size, vocab_size], [[beam_size, hidden_size], ..]
    next_char_logits, hidden_state = decoder(alive_indices, alive_hidden)
    # Adding log probabilities is equivalent to multiplying probabilities.
    # [beam_size, vocab_size]
    cumulative_logprob = (tf.expand_dims(alive_logprobs, 1) +
                          tf.nn.log_softmax(next_char_logits))

    # Pad all the finished/alive sequences so that they maintain the same shape
    # with each iteration. (A limitation of AutoGraph-generated tf.while_loops.)
    sequence_padding = tf.zeros([beam_size, max_seq_len - i - 1],
                                dtype=tf.int32)

    # Gather sequences/log probs for finished sequences
    newly_finished_sequences = tf.concat([
        alive_sequences[:, :i],
        tf.tile([[eos_token]], [beam_size, 1]),
        sequence_padding], axis=1)
    newly_finished_logprobs = cumulative_logprob[:, eos_token]
    finished_sequences, finished_logprobs = get_best_sequences(
        beam_size, finished_sequences, finished_logprobs,
        newly_finished_sequences, newly_finished_logprobs)

    # Gather sequences/log probs for alive sequences
    chosen_sequences, alive_logprobs, alive_indices = get_best_alive(
        cumulative_logprob, eos_token, beam_size, vocab_size)
    new_sequence_history = tf.gather(alive_sequences, chosen_sequences)
    # [beam_size, max_seq_len]
    alive_sequences = tf.concat([
        new_sequence_history[:, :i],
        tf.expand_dims(alive_indices, 1),
        sequence_padding], axis=1)
    alive_sequences.set_shape([beam_size, max_seq_len])
    # [[beam_size, hidden_size], ...]
    alive_hidden = tf.nest.map_structure(
        lambda s: tf.gather(s, chosen_sequences),  # pylint: disable=cell-var-from-loop
        hidden_state)

    # Exit if all alive sequences are worse than any finished sequence.
    if tf.reduce_min(finished_logprobs) > tf.reduce_max(alive_logprobs):
      break
  # Execute one final collation, just in case any of the alive sequences are
  # higher in probability than any of the finished sequences.
  finished_sequences, finished_logprobs = get_best_sequences(
      beam_size, finished_sequences, finished_logprobs,
      alive_sequences, alive_logprobs)
  return finished_sequences, finished_logprobs


class BeamSearchBenchmark(benchmark_base.ReportingBenchmark):
  """Runs benchmarks for eager/autograph variants of beam search."""

  def _get_decoder(self, vocab_size):
    return FakeDecoder(vocab_size=vocab_size)

  def _benchmark_eager(self, max_seq_len, vocab_size):

    decoder = self._get_decoder(vocab_size)

    def target():
      return beam_search(decoder, decoder.get_initial_state(1), INIT_TOKEN,
                         EOS_TOKEN, BEAM_SIZE, vocab_size,
                         max_seq_len=max_seq_len)

    self.time_execution(('Eager', max_seq_len, vocab_size),
                        target,
                        extras={
                            'max_seq_len': max_seq_len,
                            'vocab_size': vocab_size
                        })

  def _benchmark_ag(self, max_seq_len, vocab_size):

    decoder = self._get_decoder(vocab_size)
    compiled_fn = tf.function(beam_search)

    def target():
      return compiled_fn(decoder, decoder.get_initial_state(1), INIT_TOKEN,
                         EOS_TOKEN, BEAM_SIZE, vocab_size,
                         max_seq_len=max_seq_len)

    self.time_execution(('AutoGraph', max_seq_len, vocab_size),
                        target,
                        extras={
                            'max_seq_len': max_seq_len,
                            'vocab_size': vocab_size
                        })

  def benchmark_beamsearch(self):
    for max_seq_len in (10, 20, 40, 80):
      for vocab_size in (1000, 3000, 10000, 30000):
        self._benchmark_eager(max_seq_len, vocab_size)
        self._benchmark_ag(max_seq_len, vocab_size)


if __name__ == '__main__':
  tf.test.main()

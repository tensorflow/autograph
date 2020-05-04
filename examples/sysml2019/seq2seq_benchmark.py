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
"""Benchmark for a seq2seq encoder/decoder implementation.

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


BATCH_SIZE = 32
MAX_SEQ_LEN = 100
HIDDEN_SIZE = 256
VOCAB_SIZE = 1000
EMBEDDING_SIZE = 50


def seq2seq(rnn_cell, embedding, input_seq, input_seq_lengths, eos_id,
            target_seq=None, max_output_len=100):
  """An implementation of seq2seq in AutoGraph-friendly format.

  Args:
    rnn_cell: An RNNCell. Is used for both encoding and decoding.
    embedding: An embedding lookup matrix. Used for both encoding and decoding.
    input_seq: Tensor of shape [batch_size, seq_len] of vocab_ids.
    input_seq_lengths: Tensor of shape [batch_size] with sequence lengths.
    eos_id: The vocab_id corresponding to <END OF SEQUENCE>.
    target_seq: (Optional) Target sequence for teacher forcing.
    max_output_len: Maximum output length before cutting off decoder. (Otherwise
        it can run forever without emitting EOS).
  Returns:
    Tensor of shape [seq_length, batch_size, vocab_size].
  """
  batch_size = input_seq.shape[0]
  input_seq = tf.nn.embedding_lookup(embedding, input_seq)
  state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
  max_input_seq_len = tf.reduce_max(input_seq_lengths)
  # Encoder half of model
  for i in tf.range(max_input_seq_len):
    _, new_state = rnn_cell(input_seq[:, i], state)
    state = tf.where(i < input_seq_lengths, new_state, state)

  # Decoder half of model
  if target_seq is not None:
    max_output_len = target_seq.shape[1]

  outputs = tf.TensorArray(tf.float32, size=max_output_len)
  is_done = tf.zeros([batch_size], dtype=tf.bool)
  # Initial input is the end of seq token.
  eos_vector = tf.nn.embedding_lookup(embedding, tf.constant([eos_id]))
  dec_input = tf.tile(eos_vector, [batch_size, 1])
  # Run up to max_output_len steps; can exit earlier if all sequences done.
  for i in tf.range(max_output_len):
    new_output, new_state = rnn_cell(dec_input, state)
    output = tf.where(is_done, tf.zeros_like(new_output), new_output)
    outputs.write(i, output)
    if target_seq is not None:
      # if target is known, use teacher forcing
      target_word = target_seq[:, i]
    else:
      # Otherwise, pick the most likely continuation (greedy search)
      target_word = tf.argmax(output, axis=1)
    dec_input = tf.nn.embedding_lookup(embedding, target_word)
    is_done = tf.logical_or(is_done, tf.equal(target_word, eos_id))
    if tf.reduce_all(is_done):
      break
  return outputs.stack()


class Seq2SeqBenchmark(benchmark_base.ReportingBenchmark):
  """Runs benchmarks for eager/autograph/graph variants of seq2seq."""

  def _generate_fake_rnn_inputs(self,
                                batch_size=BATCH_SIZE,
                                max_seq_len=MAX_SEQ_LEN):
    np.random.seed(17)

    input_data = np.random.randint(
        0, VOCAB_SIZE, size=[batch_size, max_seq_len]).astype(np.int32)
    # Generate some varying sequence lengths but keep max(sequence_lengths)
    # a constant, for more reproducible benchmarks.
    sequence_lengths = np.concatenate(([max_seq_len],
                                       np.random.randint(
                                           max_seq_len // 2,
                                           max_seq_len,
                                           size=[batch_size - 1]))).astype(
                                               np.int32)

    for i, seq_len in enumerate(sequence_lengths):
      input_data[i, seq_len:] = 0

    input_data = tf.constant(input_data)
    sequence_lengths = tf.constant(sequence_lengths)

    return input_data, sequence_lengths

  def _create_embedding(self):
    return tf.random.uniform([VOCAB_SIZE, EMBEDDING_SIZE])

  def _create_rnn_cell(self, batch_size=BATCH_SIZE):
    rnn_cell = tf_v1.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE, dtype=tf.float32)
    rnn_cell.build(tf.TensorShape([batch_size, EMBEDDING_SIZE]))
    return rnn_cell

  def _benchmark_seq2seq(self, mode, seq2seq_variant, batch_size, max_seq_len,
                         use_teacher_forcing):
    input_seq, input_seq_lengths = self._generate_fake_rnn_inputs(
        batch_size=batch_size, max_seq_len=max_seq_len)
    target_seq, _ = self._generate_fake_rnn_inputs(
        batch_size=batch_size, max_seq_len=max_seq_len)
    rnn_cell = self._create_rnn_cell(batch_size=batch_size)
    embedding = self._create_embedding()

    if not use_teacher_forcing:
      target_seq = None

    def target():
      return seq2seq_variant(rnn_cell, embedding, input_seq,
                             input_seq_lengths, 0, target_seq=target_seq)

    self.time_execution(
        (mode, max_seq_len, batch_size, use_teacher_forcing),
        target,
        iter_volume=batch_size,
        iter_unit='examples',
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'use_teacher_forcing': use_teacher_forcing
        })

  def benchmark_seq2seq(self):
    for batch_size in (32, 64, 128):
      for max_seq_len in (64, 128):
        for use_teacher_forcing in (True, False):
          self._benchmark_seq2seq('AutoGraph', tf.function(seq2seq), batch_size,
                                  max_seq_len, use_teacher_forcing)
          self._benchmark_seq2seq('Eager', seq2seq, batch_size, max_seq_len,
                                  use_teacher_forcing)


if __name__ == '__main__':
  tf.test.main()

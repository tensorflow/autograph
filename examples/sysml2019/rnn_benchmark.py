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
"""Benchmark comparing eager, autograph, and official dynamic_rnn.

This code is tested on TF 1.13.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import benchmark_base

import numpy as np
import tensorflow as tf


tf.enable_eager_execution()


BATCH_SIZE = 32
MAX_SEQ_LEN = 100
FEATURE_SIZE = 50
HIDDEN_SIZE = 256


class RNNBenchmark(benchmark_base.ReportingBenchmark):
  """Runs benchmarks for eager/autograph/graph variants of dynamic_rnn."""

  def _generate_fake_rnn_inputs(self,
                                batch_size=BATCH_SIZE,
                                max_seq_len=MAX_SEQ_LEN):
    np.random.seed(17)

    input_data = np.random.random([batch_size, max_seq_len,
                                   FEATURE_SIZE]).astype(np.float32)
    # Generate some varying sequence lengths but keep max(sequence_lengths)
    # a constant, for more reproducible benchmarks.
    sequence_lengths = np.concatenate(([max_seq_len],
                                       np.random.randint(
                                           max_seq_len // 2,
                                           max_seq_len,
                                           size=[batch_size - 1]))).astype(
                                               np.int32)

    for i, seq_len in enumerate(sequence_lengths):
      input_data[i, seq_len:, :] = 0

    input_data = tf.constant(input_data)
    sequence_lengths = tf.constant(sequence_lengths)

    return input_data, sequence_lengths

  def _create_rnn_cell(self, batch_size=BATCH_SIZE):
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE, dtype=tf.float32)
    rnn_cell.build(tf.TensorShape([batch_size, FEATURE_SIZE]))
    return rnn_cell, rnn_cell.zero_state(batch_size, dtype=tf.float32)

  def _benchmark_eager_dynamic_rnn(self, batch_size, max_seq_len):
    input_data, sequence_lengths = self._generate_fake_rnn_inputs(
        batch_size=batch_size, max_seq_len=max_seq_len)
    rnn_cell, initial_state = self._create_rnn_cell(batch_size=batch_size)

    def eager_dynamic_rnn(rnn_cell,
                          input_data,
                          initial_state,
                          sequence_length=None):
      """An eager version of dynamic_rnn."""
      # [batch, time, features] -> [time, batch, features]
      input_data = tf.transpose(input_data, [1, 0, 2])
      outputs = []
      state = initial_state
      if sequence_length is None:
        max_seq_len = input_data.shape[0]
      else:
        max_seq_len = tf.reduce_max(sequence_length)
      for i in range(max_seq_len):
        new_output, new_state = rnn_cell(input_data[i], state)
        output = tf.where(i < sequence_length, new_output,
                          tf.zeros(new_output.shape))
        state = tf.where(i < sequence_length, new_state, state)
        outputs.append(output)
      return tf.transpose(tf.stack(outputs), [1, 0, 2]), state

    def target():
      eager_dynamic_rnn(rnn_cell, input_data, initial_state, sequence_lengths)

    self.time_execution(
        ('Eager', batch_size, max_seq_len),
        target,
        iter_volume=batch_size,
        iter_unit='examples',
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
        })

  def _benchmark_handwritten_dynamic_rnn(self, batch_size, max_seq_len):

    def my_dynamic_rnn(rnn_cell,
                       input_data,
                       initial_state,
                       sequence_length=None):
      """A handwritten reimplementation of dynamic_rnn."""
      input_data = tf.transpose(input_data, [1, 0, 2])
      outputs = tf.TensorArray(tf.float32, input_data.shape[0])
      if sequence_length is None:
        max_seq_len = input_data.shape[0]
      else:
        max_seq_len = tf.reduce_max(sequence_length)

      def while_body(i, state, outputs):
        new_output, new_state = rnn_cell(input_data[i], state)
        output = tf.where(i < sequence_length, new_output,
                          tf.zeros(new_output.shape))
        state = tf.where(i < sequence_length, new_state, state)
        outputs = outputs.write(i, output)
        return i + 1, state, outputs

      def while_cond(i, unused_state, unused_outputs):
        return i < max_seq_len

      _, state, outputs = tf.while_loop(
          while_cond,
          while_body,
          loop_vars=(tf.constant(0), initial_state, outputs))
      return tf.transpose(outputs.stack(), [1, 0, 2]), state

    with tf.Graph().as_default():
      input_data, sequence_lengths = self._generate_fake_rnn_inputs(
          batch_size=batch_size, max_seq_len=max_seq_len)
      rnn_cell, initial_state = self._create_rnn_cell(batch_size=batch_size)
      graph_output_t = my_dynamic_rnn(rnn_cell, input_data, initial_state,
                                      sequence_lengths)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def target():
          sess.run(graph_output_t)

        self.time_execution(
            ('Handwritten', batch_size, max_seq_len),
            target,
            iter_volume=batch_size,
            iter_unit='examples',
            extras={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
            })

  def benchmark_dynamic_rnn(self):
    for batch_size in (32, 64, 128):
      for max_seq_len in (64, 128):
        self._benchmark_eager_dynamic_rnn(batch_size, max_seq_len)
        self._benchmark_handwritten_dynamic_rnn(batch_size, max_seq_len)
        self._benchmark_ag_dynamic_rnn(batch_size, max_seq_len)
        self._benchmark_official_dynamic_rnn(batch_size, max_seq_len)

  def _benchmark_ag_dynamic_rnn(self, batch_size, max_seq_len):

    def ag_dynamic_rnn(rnn_cell,
                       input_data,
                       initial_state,
                       sequence_length=None):
      """An autograph-able reimplementation of subset of dynamic_rnn."""
      # [batch, time, features] -> [time, batch, features]
      input_data = tf.transpose(input_data, [1, 0, 2])
      if sequence_length is None:
        max_seq_len = input_data.shape[0]
      else:
        max_seq_len = tf.reduce_max(sequence_length)

      outputs = tf.TensorArray(tf.float32, size=max_seq_len)
      state = initial_state
      for i in tf.range(max_seq_len):
        new_output, new_state = rnn_cell(input_data[i], state)
        output = tf.where(i < sequence_length, new_output,
                          tf.zeros(new_output.shape))
        state = tf.where(i < sequence_length, new_state, state)
        outputs = outputs.write(i, output)
      return tf.transpose(outputs.stack(), [1, 0, 2]), state

    ag_dynamic_rnn = tf.autograph.to_graph(ag_dynamic_rnn)

    with tf.Graph().as_default():
      input_data, sequence_lengths = self._generate_fake_rnn_inputs(
          batch_size=batch_size, max_seq_len=max_seq_len)
      rnn_cell, initial_state = self._create_rnn_cell(batch_size=batch_size)
      rnn_output = ag_dynamic_rnn(rnn_cell, input_data, initial_state,
                                  sequence_lengths)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def target():
          sess.run(rnn_output)

        self.time_execution(
            ('AutoGraph', batch_size, max_seq_len),
            target,
            iter_volume=batch_size,
            iter_unit='examples',
            extras={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
            })

  def _benchmark_official_dynamic_rnn(self, batch_size, max_seq_len):
    with tf.Graph().as_default():
      input_data, sequence_lengths = self._generate_fake_rnn_inputs(
          batch_size=batch_size, max_seq_len=max_seq_len)
      rnn_cell, initial_state = self._create_rnn_cell(batch_size=batch_size)

      rnn_output = tf.nn.dynamic_rnn(
          rnn_cell,
          input_data,
          initial_state=initial_state,
          sequence_length=sequence_lengths)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def target():
          sess.run(rnn_output)

        self.time_execution(
            ('tf.nn.dynamic_rnn', batch_size, max_seq_len),
            target,
            iter_volume=batch_size,
            iter_unit='examples',
            extras={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
            })


if __name__ == '__main__':
  tf.test.main()

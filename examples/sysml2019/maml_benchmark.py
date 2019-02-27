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
"""Benchmark for a basic MAML implementation.

Based on implementation found at https://github.com/cbfinn/maml.

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




AMP_RANGE = (0.1, 5.0)
PHASE_RANGE = (0.0, np.pi)
INPUT_RANGE = (-5.0, 5.0)
INPUT_SIZE = 1
OUTPUT_SIZE = 1
LOCAL_TRAIN_EXAMPLES = 10
LOCAL_VALID_EXAMPLES = 10
META_BATCH_SIZE = 25

HIDDEN_SIZES = (40, 40)

LOCAL_LEARNING_RATE = 1e-3
LOCAL_LEARNING_STEPS = 1
LOCAL_LEARNING_STEPS_TEST = 5
TRAIN_SAMPES_SLICE = slice(0, LOCAL_TRAIN_EXAMPLES)
VALID_SAMPES_SLICE = slice(LOCAL_TRAIN_EXAMPLES,
                           LOCAL_TRAIN_EXAMPLES + LOCAL_VALID_EXAMPLES)

META_LR = 0.001


def sin_dataset():
  """Procedural dataset that generates samples for the sin function."""
  assert INPUT_SIZE == OUTPUT_SIZE

  total_samples = LOCAL_TRAIN_EXAMPLES + LOCAL_VALID_EXAMPLES

  def gen_input_outputs(_):
    amp = tf.random.uniform((), AMP_RANGE[0], AMP_RANGE[1])
    phase = tf.random.uniform((), PHASE_RANGE[0], PHASE_RANGE[1])
    init_inputs = tf.random.uniform((total_samples, INPUT_SIZE), INPUT_RANGE[0],
                                    INPUT_RANGE[1])
    outputs = amp * tf.math.sin(init_inputs - phase)
    return init_inputs, outputs, amp, phase

  ds = tf.data.Dataset.range(2)
  ds = ds.map(gen_input_outputs)

  return ds


def w(i):
  return 'w{}'.format(i)


def b(i):
  return 'b{}'.format(i)


def model_weights():
  weights = {}
  aug_sizes = (INPUT_SIZE,) + HIDDEN_SIZES + (OUTPUT_SIZE,)
  for i in range(1, len(aug_sizes)):
    weights[w(i)] = tf.Variable(
        tf.random.truncated_normal((aug_sizes[i - 1], aug_sizes[i]),
                                   stddev=0.01))
    weights[b(i)] = tf.Variable(tf.zeros((aug_sizes[i],)))
  return weights


def model(inputs, weights):
  x = inputs
  for i in range(1, len(HIDDEN_SIZES) + 1):
    x = tf.nn.relu(tf.matmul(x, weights[w(i)]) + weights[b(i)])
  i = len(HIDDEN_SIZES) + 1
  return tf.matmul(x, weights[w(i)]) + weights[b(i)]


def mse(preds, labels):
  return tf.reduce_mean(tf.square(preds - labels))


def local_learn(inputs, outputs, weights, num_steps):
  """Runs a classical training loop."""
  learned_weights = tf.nest.map_structure(lambda w: w.read_value(), weights)

  for _ in tf.range(num_steps):
    # Inference
    with tf.GradientTape() as tape:
      tf.nest.map_structure(tape.watch, learned_weights)
      y_pred = model(inputs, learned_weights)
      step_loss = mse(y_pred, outputs)

    # SGD step
    grads = tape.gradient(step_loss, learned_weights)
    learned_weights = tf.nest.map_structure(
        lambda w, g: w - LOCAL_LEARNING_RATE * g, learned_weights, grads)

  return learned_weights


def metalearn(weights, opt, meta_steps):
  """Runs a MAML learning loop."""
  ds = sin_dataset().repeat().batch(META_BATCH_SIZE).take(meta_steps)

  for inputs, outputs, _, _ in ds:
    train_inputs = inputs[:, TRAIN_SAMPES_SLICE, :]
    valid_inputs = inputs[:, VALID_SAMPES_SLICE, :]
    train_outputs = outputs[:, TRAIN_SAMPES_SLICE, :]
    valid_outputs = outputs[:, VALID_SAMPES_SLICE, :]

    with tf.GradientTape() as tape:
      tf.nest.map_structure(tape.watch, weights)

      # Per-task learning
      task_losses = tf.TensorArray(tf.float32, size=META_BATCH_SIZE)
      for i in tf.range(META_BATCH_SIZE):
        # Train on the training data points
        learned_weights = local_learn(train_inputs[i], train_outputs[i],
                                      weights, LOCAL_LEARNING_STEPS)
        # Calucalate loss on the validation data points
        learned_valid_outputs = model(valid_inputs[i], learned_weights)
        # Use the validation error for meta training
        task_loss = mse(learned_valid_outputs, valid_outputs[i])
        task_losses = task_losses.write(i, task_loss)

      # Average per-task validation errors.
      meta_loss = tf.reduce_mean(task_losses.stack())

    # Take a single meta-training step.
    meta_grads = tape.gradient(meta_loss, weights)
    grads_and_vars = zip(tf.nest.flatten(meta_grads), tf.nest.flatten(weights))
    opt.apply_gradients(grads_and_vars)


class MAMLBenchmark(benchmark_base.ReportingBenchmark):
  """Basic benchmark for the MAML example model."""

  def _run_benchmark(self, name, metalearn_function, meta_steps):
    init_weights = model_weights()
    opt = tf.keras.optimizers.Adam()

    def target():
      metalearn_function(init_weights, opt, meta_steps)

    self.time_execution((name, meta_steps),
                        target,
                        extras={
                            'meta_steps': meta_steps,
                        })

  def benchmark_maml(self):
    # TODO(mdanatg): Remove this override.
    all_current_features = tf.autograph.experimental.Feature.all_but(
        tf.autograph.experimental.Feature.AUTO_CONTROL_DEPS)

    for meta_steps in (1, 10):
      self._run_benchmark('Eager', metalearn, meta_steps)
      metalearn_autograph = tf.function(
          metalearn, experimental_autograph_options=all_current_features)
      self._run_benchmark('AutoGraph', metalearn_autograph, meta_steps)


if __name__ == '__main__':
  tf.test.main()

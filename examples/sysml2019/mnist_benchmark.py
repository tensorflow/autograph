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
"""Benchmark comparing MNIST with eager mode, graph mode, and autograph.

This code assumes TF 1.X.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import benchmark_base

import tensorflow as tf


tf.enable_eager_execution()


def get_data_and_params():
  """Set up input dataset and variables."""
  (train_x, train_y), _ = tf.keras.datasets.mnist.load_data()
  tf.set_random_seed(0)
  hparams = tf.contrib.training.HParams(
      batch_size=200,
      learning_rate=0.1,
      train_steps=101,
  )
  dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
  dataset = dataset.repeat()
  dataset = dataset.shuffle(hparams.batch_size * 10)
  dataset = dataset.batch(hparams.batch_size)

  def reshape_ex(x, y):
    return (tf.to_float(tf.reshape(x, (-1, 28 * 28))) / 256.0,
            tf.one_hot(tf.squeeze(y), 10))

  dataset = dataset.map(reshape_ex)
  w = tf.get_variable('w0', (28 * 28, 10))
  b = tf.get_variable('b0', (10,), initializer=tf.zeros_initializer())
  opt = tf.train.GradientDescentOptimizer(hparams.learning_rate)
  return dataset, opt, hparams, w, b


def model_fn(x, w, b):
  return tf.matmul(x, w) + b


def loss_fn(x, y, w, b):
  y_ = model_fn(x, w, b)
  return tf.losses.softmax_cross_entropy(y, y_)


class MNISTBenchmark(benchmark_base.ReportingBenchmark):
  """Benchmark a simple model training loop on MNIST digits dataset."""

  def benchmark_eager(self):
    ds, opt, hp, w, b = get_data_and_params()
    iterator = iter(ds)

    def target():
      """Eager implementation of training loop."""
      for i, (x, y) in enumerate(iterator):
        if i >= hp.train_steps:
          break
        with tf.contrib.eager.GradientTape() as tape:
          tape.watch(w)
          tape.watch(b)
          loss_val = loss_fn(x, y, w, b)
        dw, db = tape.gradient(loss_val, (w, b))
        opt.apply_gradients(((dw, w), (db, b)))
        if i % 100 == 0:
          print('Step', i, ':', loss_val)
      assert 0.1 < loss_val < 1, loss_val

    self.time_execution(
        'Eager',
        target,
        iter_volume=hp.train_steps,
        iter_unit='training steps')

  def benchmark_legacy_tf(self, loss_fun=loss_fn):
    with tf.Graph().as_default():
      ds, opt, hp, w, b = get_data_and_params()
      x, y = ds.make_one_shot_iterator().get_next()
      loss_t = loss_fn(x, y, w, b)
      train_op = opt.minimize(loss_t, var_list=(w, b))
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def target():
          for i in range(hp.train_steps):
            loss_val, _ = sess.run([loss_t, train_op])
            if i % 100 == 0:
              print('Step', i, ':', loss_val)
          assert 0.1 < loss_val < 1, loss_val

        self.time_execution(
            'Classical',
            target,
            iter_volume=hp.train_steps,
            iter_unit='training steps')

  def benchmark_autograph(self):

    def loop(ds, opt, hp, w, b):
      """AG implementation of training loop."""
      loss = 0.0
      iterator = ds.make_one_shot_iterator()
      # TODO(brianklee): Rewrite with only one usage of iterator.get_next().
      # Currently needs two calls because of the control_dependencies clause.
      # See b/109924949, b/117497661
      x, y = iterator.get_next()
      for i in tf.range(hp.train_steps):
        loss = loss_fn(x, y, w, b)
        if i % 100 == 0:
          print('Step', i, ':', loss)
        with tf.control_dependencies([opt.minimize(loss, var_list=(w, b))]):
          # This ensures that each iteration of the loop has a dependency
          # on the previous iteration completing. Otherwise you get async SGD.
          x, y = iterator.get_next()

      return loss

    loop = tf.autograph.to_graph(loop)

    with tf.Graph().as_default():
      ds, opt, hp, w, b = get_data_and_params()
      loss = loop(ds, opt, hp, w, b)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def target():
          loss_val = sess.run(loss)
          assert 0.1 < loss_val < 1, loss_val

        self.time_execution(
            'AutoGraph',
            target,
            iter_volume=hp.train_steps,
            iter_unit='training steps')

  def benchmark_handwritten(self):
    with tf.Graph().as_default():
      ds, opt, hp, w, b = get_data_and_params()
      iterator = ds.make_one_shot_iterator()

      def loop_body(i, unused_previous_loss_t):
        """Manual implementation of training loop."""
        # Call get_next() inside body or else training happens repeatedly on
        # the first minibatch only.
        x, y = iterator.get_next()
        loss_t = loss_fn(x, y, w, b)
        train_op = opt.minimize(loss_t, var_list=(w, b))
        i = tf.cond(tf.equal(i % 100, 0),
                    lambda: tf.Print(i, [i, loss_t], message='Step, loss: '),
                    lambda: i)

        with tf.control_dependencies([train_op]):
          return i + 1, loss_t

      _, final_loss_t = tf.while_loop(
          lambda i, _: i < hp.train_steps,
          loop_body,
          [tf.constant(0), tf.constant(0.0)])

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def target():
          loss_val = sess.run(final_loss_t)
          assert 0.1 < loss_val < 1, loss_val

        self.time_execution(
            'Handwritten',
            target,
            iter_volume=hp.train_steps,
            iter_unit='training steps')


if __name__ == '__main__':
  tf.test.main()

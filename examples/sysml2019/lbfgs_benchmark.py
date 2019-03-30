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
"""Benchmark for a basic L-BFGS implementation without beam search.

Adapted from
https://github.com/yaroslavvb/stuff/blob/master/eager_lbfgs/eager_lbfgs.py.

This code requires TF 2.0 or newer.
Use `pip install tf-nightly-2.0-preview` to install.
Tip: Run the `pip install` command in a separate virtual environment
(Virtualenv, Anaconda) to avoid clobbering an existing TF installation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import benchmark_base

import tensorflow as tf




INPUT_SIZE = 28 * 28
BATCH_SIZE = 100

MAX_ITER = 20
MAX_EVAL = 25
TOL_F = 1e-5
TOL_X = 1e-9
N_CORRECTIONS = 100
LEARNING_RATE = 1.0


def mnist_dataset():
  """Loads the MNIST dataset."""

  def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.reshape(x, (INPUT_SIZE,))
    y = tf.cast(y, tf.int64)
    return x, y

  (x, y), _ = tf.keras.datasets.mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.take(BATCH_SIZE).batch(BATCH_SIZE)
  return ds


def cb_allocate(el, size):
  """Basic primitive to allocate a circular buffer."""
  # TODO(mdanatg): Rewrite as a namedtuple.
  el = tf.convert_to_tensor(el)
  buff = tf.TensorArray(
      el.dtype,
      size=size + 1,
      element_shape=tf.TensorShape(None),
      clear_after_read=False)
  for i in tf.range(size):
    buff = buff.write(i, el)
  begin = 0
  end = 0
  return buff, begin, end


def cb_append(buff, begin, end, size, el):
  """Circular buffer append primitive."""
  buff = buff.write(end, el)
  end = (end + 1) % size
  if tf.equal(end, begin):
    begin = (begin + 1) % size
  return buff, begin, end


def cb_len(begin, end, size):
  """Circular buffer length primitive."""
  return (end - begin) % size


def cb_range(begin, end, size):
  """Circular buffer range primitive."""
  if end < begin:
    virtual_end = end + size
  else:
    virtual_end = end
  return tf.range(begin, virtual_end) % size


def cb_rev_range(begin, end, size):
  """Circular buffer reversed range primitive."""
  if end < begin:
    virtual_end = end + size
  else:
    virtual_end = end
  return tf.range(virtual_end - 1, begin - 1, -1) % size


def dot(a, b):
  return tf.reduce_sum(a * b)


def loss_fn(w_flat, data):
  w = tf.reshape(w_flat, [INPUT_SIZE, -1])
  x = tf.matmul(data, w)
  x = tf.sigmoid(x)
  x = tf.matmul(x, w, transpose_b=True)
  x = tf.sigmoid(x)
  return tf.reduce_mean(tf.square(x - data))


def loss_and_grad(w_flat, data):
  with tf.GradientTape() as g:
    g.watch(w_flat)
    f = loss_fn(w_flat, data)
  g = g.gradient(f, w_flat)
  return f, g


def lbfgs_eager(x, data):
  """Implementation of L-BFGS in AutoGraph / TF Eager."""

  f, g = loss_and_grad(x, data)
  f_hist = tf.TensorArray(f.dtype, size=0, dynamic_size=True)
  f_hist = f_hist.write(0, f)
  f_evals = 1

  # Check optimality of initial point.
  if tf.reduce_sum(tf.abs(g)) <= TOL_F:
    tf.print('Optimality condition below TOL_F.')
    return x, f_hist.stack()

  # Pre-allocate some buffers.
  dirs_buff, dirs_begin, dirs_end = cb_allocate(tf.zeros_like(g), N_CORRECTIONS)
  steps_buff, steps_begin, steps_end = cb_allocate(
      tf.zeros_like(g), N_CORRECTIONS)
  ro, _, _ = cb_allocate(0.0, N_CORRECTIONS)
  al, _, _ = cb_allocate(0.0, N_CORRECTIONS)

  n_iter = tf.constant(0)
  d = -g
  prev_g = g
  h_diag = 1.0
  t = tf.minimum(1.0, 1.0 / tf.reduce_sum(tf.abs(g)))

  while n_iter <= MAX_ITER:
    n_iter += 1

    if n_iter > 1:
      y = g - prev_g
      s = d * t
      ys = dot(y, s)

      if ys > 1e-10:
        dirs_buff, dirs_begin, dirs_end = cb_append(dirs_buff, dirs_begin,
                                                    dirs_end, N_CORRECTIONS, s)
        steps_buff, steps_begin, steps_end = cb_append(
            steps_buff, steps_begin, steps_end, N_CORRECTIONS, y)
        h_diag = ys / dot(y, y)

      # Approximate inverse Hessian-gradient product.
      q = -g
      for i in cb_rev_range(dirs_begin, dirs_end, N_CORRECTIONS):
        ro = ro.write(i, 1 / dot(steps_buff.read(i), dirs_buff.read(i)))
        al = al.write(i, dot(dirs_buff.read(i), q) * ro.read(i))
        q = q - al.read(i) * steps_buff.read(i)

      r = q * h_diag
      for i in cb_range(dirs_begin, dirs_end, N_CORRECTIONS):
        be = dot(steps_buff.read(i), r) * ro.read(i)
        r += (al.read(i) - be) * dirs_buff.read(i)

      d = r

    prev_g = g
    prev_f = f

    # Step direction (directional derivative).
    gtd = dot(g, d)

    if gtd > -TOL_X:
      tf.print('Can not make progress along direction.')
      break

    # Step size
    if n_iter > 1:
      t = LEARNING_RATE

    # No line search, simply move with fixed step.
    x += t * d

    if n_iter < MAX_ITER:
      # Skip re-evaluation after last iteration.
      f, g = loss_and_grad(x, data)
      f_evals += 1  # This becomes less trivial when using line search.

      f_hist = f_hist.write(f_hist.size(), f)

    # Check conditions, again on all-but-final-eval only.
    if tf.equal(n_iter, MAX_ITER):
      break

    if f_evals >= MAX_EVAL:
      tf.print('Max number of function evals.')
      break

    if tf.reduce_sum(tf.abs(d * t)) <= TOL_X:
      tf.print('Step size below TOL_X.')
      break

    f_delta = tf.abs(f - prev_f)
    if f_delta < TOL_X:
      tf.print('Function value changing less than TOL_X.', f_delta)
      break

  return x, f_hist.stack()


lbfgs_autograph = tf.function(
    lbfgs_eager,
    experimental_autograph_options=tf.autograph.experimental.Feature.ALL)


class LBFGSBenchmark(benchmark_base.ReportingBenchmark):
  """Basic benchmark for the L-BFGS algorithm."""

  def _run_benchmark(self, name, algorithm_function, hidden_size, data):
    w_flat = tf.Variable(tf.zeros((INPUT_SIZE * hidden_size,)))

    def target():
      new_w_flat, _ = algorithm_function(w_flat.read_value(), data)
      _ = new_w_flat.numpy()

    self.time_execution((name, hidden_size),
                        target,
                        extras={
                            'hidden_size': hidden_size,
                        })

  def benchmark_lbfgs(self):
    data, _ = next(iter(mnist_dataset()))
    # TODO(mdanatg): Use more interesting parametrizations.
    # TODO(mdanatg): Double check correctness.
    for hidden_size in (100,):
      self._run_benchmark('Eager', lbfgs_eager, hidden_size, data)
      self._run_benchmark('AutoGraph', lbfgs_autograph, hidden_size, data)


if __name__ == '__main__':
  tf.test.main()

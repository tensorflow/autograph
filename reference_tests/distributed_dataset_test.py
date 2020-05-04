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
"""Tests involving the tf.distributed datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import tensorflow.compat.v2 as tf


tf.enable_v2_behavior()


def dataset_no_vars_loop(ds, dds):
  for pr in dds:
    tf.print(ds.reduce('SUM', pr, axis=None))


def iterator_no_vars_loop(ds, dds):
  for pr in iter(dds):
    tf.print(ds.reduce('SUM', pr, axis=None))


def dataset_single_var_loop(ds, dds):
  s = 0
  for pr in dds:
    # TODO(mdan): It would be nice to be able to write s = s * 10 + pr.
    s = s * 10 + ds.reduce('SUM', pr, axis=None)
    # TODO(mdan): This looks like a bug.
    s.set_shape(())
  return s


def iterator_single_var_loop(ds, dds):
  s = 0
  for pr in iter(dds):
    s = s * 10 + ds.reduce('SUM', pr, axis=None)
  return s


def dataset_two_vars_loop(ds, dds):
  s = 0
  p = 1
  for pr in dds:
    e = ds.reduce('SUM', pr, axis=None)
    e.set_shape(())
    s += e
    p *= e
  return s, p


def iterator_two_vars_loop(ds, dds):
  s = 0
  p = 1
  for pr in iter(dds):
    e = ds.reduce('SUM', pr, axis=None)
    e.set_shape(())
    s += e
    p *= e
  return s, p


def dataset_enumeration(ds, dds):
  s = 0
  p = 1
  for i, pr in enumerate(dds):
    e = ds.reduce('SUM', pr, axis=None)
    e.set_shape(())
    s = s * 10 + e
    p *= i
  return s, p


def iterator_next(ds, dds):
  itr = iter(dds)
  return ds.reduce('SUM', next(itr), axis=None)


def iterator_next_multiple_calls(ds, dds):
  itr = iter(dds)
  a = ds.reduce('SUM', next(itr), axis=None)
  b = ds.reduce('SUM', next(itr), axis=None)
  return a * 10 + b


def iterator_next_in_limited_loop(ds, dds, n):
  itr = iter(dds)
  s = 0
  for _ in range(n):
    s = s * 10 + ds.reduce('SUM', next(itr), axis=None)
  return s


def iterator_next_stopping(ds, dds, cond):
  # This case will raise, but not the expected StopIteration error.
  itr = iter(dds)
  while cond:
    ds.reduce('SUM', next(itr), axis=None)


def iterator_next_with_catching_stop_iteration(ds, dds, cond):
  # This is the one instance when the use of TF iterators does not work as
  # intended. In graph mode, the `except` below will never catch, and the
  # tf.function will raise the error instead.
  # TODO(b/132311724): The error should be friendlier here.
  # Note: b/132298783 covers actually supporting this pattern.
  itr = iter(dds)
  try:
    while cond:
      ds.reduce('SUM', next(itr), axis=None)
  except StopIteration:
    pass


class ReferenceTest(reference_test_base.TestCase):

  def setUp(self):
    super(ReferenceTest, self).setUp()
    cpus = tf.config.experimental.list_physical_devices('CPU')
    tf.config.experimental.set_virtual_device_configuration(
        cpus[0], [tf.config.experimental.VirtualDeviceConfiguration()] * 2)

    strategy = tf.distribute.MirroredStrategy()
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.reshape(tf.range(40), (10, 4)))

    self.ds = strategy
    self.dds = strategy.experimental_distribute_dataset(dataset)

  def test_dataset_no_vars_loop(self):
    self.assertFunctionMatchesEager(dataset_no_vars_loop, self.ds, self.dds)

  def test_iterator_no_vars_loop(self):
    with self.assertRaises(RuntimeError):
      tf.function(iterator_no_vars_loop)(self.ds, self.dds)

  def test_dataset_single_var_loop(self):
    self.assertFunctionMatchesEager(dataset_single_var_loop, self.ds, self.dds)

  def test_iterator_single_var_loop(self):
    with self.assertRaises(RuntimeError):
      tf.function(iterator_single_var_loop)(self.ds, self.dds)

  def test_dataset_two_vars_loop(self):
    self.assertFunctionMatchesEager(dataset_two_vars_loop, self.ds, self.dds)

  def test_iterator_two_vars_loop(self):
    with self.assertRaises(RuntimeError):
      tf.function(iterator_two_vars_loop)(self.ds, self.dds)

  def test_iterator_next(self):
    self.assertFunctionMatchesEager(iterator_next, self.ds, self.dds)

  def test_iterator_next_multiple_calls(self):
    self.assertFunctionMatchesEager(iterator_next_multiple_calls, self.ds,
                                    self.dds)

  def test_iterator_next_in_limited_loop(self):
    self.assertFunctionMatchesEager(iterator_next_in_limited_loop, self.ds,
                                    self.dds, 0)
    self.assertFunctionMatchesEager(iterator_next_in_limited_loop, self.ds,
                                    self.dds, 1)
    self.assertFunctionMatchesEager(iterator_next_in_limited_loop, self.ds,
                                    self.dds, 3)

  def test_iterator_next_stopping(self):
    with self.assertRaises(tf.errors.OutOfRangeError):
      tf.function(iterator_next_stopping)(self.ds, self.dds, tf.constant(True))

  def test_iterator_next_with_catching_stop_iteration(self):
    with self.assertRaises(tf.errors.OutOfRangeError):
      tf.function(iterator_next_with_catching_stop_iteration)(self.ds, self.dds,
                                                              tf.constant(True))


if __name__ == '__main__':
  tf.test.main()

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
"""Simple call to a print function preceding other computations.

The call may be wrapped inside a py_func, but tf.Print should be used if
possible. The subsequent computations will be gated by the print function
execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import numpy as np
import tensorflow.compat.v1 as tf


# TODO(mdan): Allow functions that do not have a return value.
# (either that or raise an error if a return value is missing)


def lone_print(x):
  print(x)
  return x + 1


def print_multiple_values(x):
  print('x is', x)
  return x + 1


def multiple_prints(x, y):
  print('x is', x)
  print('y is', y)
  return x + 1


def print_with_nontf_values(x):
  print('x is', x, {'foo': 'bar'})
  return x + 1


def print_in_cond(x):
  if x == 0:
    print(x)
  return x


def tf_print(x):
  tf.print(x)
  return x + 1


class ReferenceTest(reference_test_base.TestCase):

  def test_lone_print(self):
    self.assertNativeMatchesCompiled(lone_print, 1)
    self.assertNativeMatchesCompiled(lone_print, np.array([1, 2, 3]))

  def test_print_multiple_values(self):
    self.assertNativeMatchesCompiled(print_multiple_values, 1)
    self.assertNativeMatchesCompiled(print_multiple_values, np.array([1, 2, 3]))

  def test_multiple_prints(self):
    self.assertNativeMatchesCompiled(multiple_prints, 1, 2)
    self.assertNativeMatchesCompiled(multiple_prints, np.array([1, 2, 3]), 4)

  def test_print_with_nontf_values(self):
    self.assertNativeMatchesCompiled(print_with_nontf_values, 1)
    self.assertNativeMatchesCompiled(print_with_nontf_values,
                                     np.array([1, 2, 3]))

  def test_print_in_cond(self):
    self.assertNativeMatchesCompiled(print_in_cond, 0)
    self.assertNativeMatchesCompiled(print_in_cond, 1)

  def test_tf_print(self):
    self.assertTfMatchesCompiled(tf_print, 0)


if __name__ == '__main__':
  tf.test.main()

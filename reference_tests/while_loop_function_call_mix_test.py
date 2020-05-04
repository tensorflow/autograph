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
"""While loops mixed with function calls."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import tensorflow.compat.v1 as tf


def basic_fn(x):
  return x * 2


def function_call_inside_cond(n):
  i = 0
  s = 0
  while i < basic_fn(n):
    s += i
    i += 1
  return s


def function_call_inside_body(n):
  i = 0
  s = 0
  while i < n:
    s += basic_fn(i)
    i += 1
  return s


def print_inside_body(n):
  i = 0
  s = 0
  while i < n:
    s += i
    print(s)
    i += 1
  return s


class ReferenceTest(reference_test_base.TestCase):
  """Base class for the reference tests."""

  def setUp(self):
    super(ReferenceTest, self).setUp()
    self.convert = reference_test_base.tf_function_custom(
        tf.autograph.experimental.Feature.all_but(
            tf.autograph.experimental.Feature.AUTO_CONTROL_DEPS))

  def test_basic(self):
    self.assertNativeMatchesCompiled(function_call_inside_cond, 3)
    self.assertNativeMatchesCompiled(function_call_inside_body, 3)
    self.assertNativeMatchesCompiled(print_inside_body, 3)


if __name__ == '__main__':
  tf.test.main()

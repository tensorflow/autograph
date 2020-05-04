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
"""Calls to dynamic functions.

Dynamic functions include:
 * function variables
 * function parameters
 * factories
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import tensorflow.compat.v1 as tf


def function_1(x):
  return x * x * x


def function_2(x):
  return -1 * x + 11


def factory():
  return function_1


def factory_dynamic_fn(x):
  f = factory()
  return f(x)


def param_dynamic_fn(f, x):
  return f(x)


def variable_dynamic_fn(x):
  f = function_1
  return f(x)


def variable_dynamic_whitelisted_fn(x):
  f = tf.identity
  return f(x)


def dynamic_fn_with_kwargs(f, x):
  return f(x=x)


class ReferenceTest(reference_test_base.TestCase):

  def test_basic(self):
    self.assertNativeMatchesCompiled(factory_dynamic_fn, 1)
    self.assertNativeMatchesCompiled(param_dynamic_fn, function_1, 1)
    self.assertNativeMatchesCompiled(variable_dynamic_fn, 1)
    self.assertTfMatchesCompiled(variable_dynamic_whitelisted_fn, 1)
    self.assertTfMatchesCompiled(dynamic_fn_with_kwargs, function_1, 1)


if __name__ == '__main__':
  tf.test.main()

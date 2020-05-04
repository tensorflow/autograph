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
"""Simple call to a builtin function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import reference_test_base
import mock
import six
import tensorflow.compat.v1 as tf


# TODO(mdan): Add tests for all builtins.


def xrange_call(x):
  return xrange(x)


def dict_call(x):
  return dict(foo=x)


def dict_call_aliased(x):
  def fake_dict(x):
    return x

  dict = fake_dict  # pylint:disable=redefined-builtin
  return dict(x)


def dict_call_dynamic(x):
  def gen_dict():
    return dict

  d = gen_dict()
  return d(foo=x)


def len_call(x):
  return len(x)


def nested_call(x):
  return list(range(len(x)))


def len_call_aliased(x):

  def fake_len(x):
    return x

  len = fake_len  # pylint:disable=redefined-builtin
  return len(x)


def len_call_dynamic(x):

  def gen_len():
    return len

  l = gen_len()
  return l(x)


def len_call_on_mock():
  x = mock.MagicMock()
  return len(x)


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ('to_graph', reference_test_base.to_graph),
      ('to_graph_nonrecursive', reference_test_base.to_graph_nonrecursive),
  ])
  def test_basic(self, conversion_func):
    self.convert = conversion_func
    self.assertNativeMatchesCompiled(dict_call, 1)
    self.assertNativeMatchesCompiled(len_call, [1, 2])
    self.assertNativeMatchesCompiled(dict_call_aliased, 1)
    self.assertNativeMatchesCompiled(len_call_aliased, [1, 2])
    self.assertNativeMatchesCompiled(dict_call_dynamic, 1)
    self.assertNativeMatchesCompiled(len_call_dynamic, [1, 2])
    self.assertNativeMatchesCompiled(nested_call, [1, 2, 3])
    self.assertNativeMatchesCompiled(nested_call, [1, 2, 3])
    if six.PY2:
      self.assertNativeMatchesCompiled(xrange_call, 3)


if __name__ == '__main__':
  tf.test.main()

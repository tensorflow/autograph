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
"""Composite names (attributes) in control flow.

Generally, composite symbols should be treated like regular ones.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import tensorflow.compat.v1 as tf


def cond_basic(x):
  if x.a > 0:
    x.b = 1
  else:
    x.b = -1
  return x


def while_basic(x, y):
  while x > 0:
    x -= 1
    y.a += 1
  return y


def while_state_only(y):
  while y.b <= 10:
    y.a += 1
    y.b *= 2
  return y


def for_basic(n, x, y):
  for i in range(n):
    x -= 1
    y.a += i
  return y


def for_state_only(n, y):
  for _ in range(n):
    y.a += 1
  return y


# TODO(mdan): More tests needed. Many pitfalls around mutating objects this way.


class ReferenceTest(reference_test_base.TestCase):

  def test_cond_basic(self):
    self.assertNativeMatchesCompiled(
        cond_basic,
        reference_test_base.MutableContainer(a=1, b=0),
    )
    self.assertNativeMatchesCompiled(
        cond_basic,
        reference_test_base.MutableContainer(a=0, b=0),
    )

  def test_while_basic(self):
    self.assertNativeMatchesCompiled(
        while_basic,
        3,
        reference_test_base.MutableContainer(a=3, b=0),
    )
    self.assertNativeMatchesCompiled(
        while_basic,
        0,
        reference_test_base.MutableContainer(a=7, b=0),
    )

  def test_while_state_only(self):
    self.assertNativeMatchesCompiled(
        while_state_only,
        reference_test_base.MutableContainer(a=3, b=1),
    )
    self.assertNativeMatchesCompiled(
        while_state_only,
        reference_test_base.MutableContainer(a=7, b=10),
    )

  def test_for_basic(self):
    self.assertNativeMatchesCompiled(
        for_basic,
        5,
        3,
        reference_test_base.MutableContainer(a=3, b=0),
    )
    self.assertNativeMatchesCompiled(
        for_basic,
        5,
        0,
        reference_test_base.MutableContainer(a=7, b=0),
    )

  def test_for_state_only(self):
    self.assertNativeMatchesCompiled(
        for_state_only,
        5,
        reference_test_base.MutableContainer(a=3, b=0),
    )
    self.assertNativeMatchesCompiled(
        for_state_only,
        0,
        reference_test_base.MutableContainer(a=7, b=0),
    )


if __name__ == '__main__':
  tf.test.main()

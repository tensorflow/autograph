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
"""Basic assertions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import tensorflow.compat.v1 as tf


def simple_assertion(x):
  assert x > 0
  return x


class ReferenceTest(reference_test_base.TestCase):

  def test_basic(self):
    self.assertNativeMatchesCompiled(simple_assertion, 1)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.try_execute_compiled(simple_assertion, 0)


if __name__ == '__main__':
  tf.test.main()

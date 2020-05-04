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
"""Loops with the exotic else construct."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import tensorflow.compat.v2 as tf


tf.enable_v2_behavior()


def for_else(l1, l2):
  s = 0
  for c in l1:
    if c in l2:
      break
    s = s * 10 + c
  else:
    s = -1000
  return s


def while_else(x, y):
  s = 0
  while x > 0:
    x -= 1
    if x > y:
      break
    s += x
  else:
    s = -100
  return s


class LoopControlFlowTest(reference_test_base.TestCase):

  def test_for_else(self):
    with self.assertRaisesRegex(NotImplementedError, 'for/else'):
      tf.function(for_else)([], [])

  def test_while_else(self):
    with self.assertRaisesRegex(NotImplementedError, 'while/else'):
      tf.function(while_else)(0, 0)


if __name__ == '__main__':
  tf.test.main()

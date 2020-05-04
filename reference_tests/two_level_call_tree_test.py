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
"""Call to a second user function.

The second function will be converted as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import tensorflow.compat.v1 as tf


def f1(x):
  return x + 1


def f2(x):
  return 2 * f1(x)


class ReferenceTest(reference_test_base.TestCase):

  def test_basic(self):
    self.assertNativeMatchesCompiled(f1, 1)
    self.assertNativeMatchesCompiled(f2, 1)


if __name__ == '__main__':
  tf.test.main()

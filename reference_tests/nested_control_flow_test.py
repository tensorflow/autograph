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
"""Nested loops and conditional statements (e.g. while, for, if).

Meant to verify that arbitrarily nested statements are processed correctly.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import tensorflow.compat.v1 as tf


def independent_ifs(x, y):
  z = 0
  if x > 0:
    if y > 0:
      z = x + y
  return z


def dependent_inner_if(x):
  y = 0
  if x > 0:
    y = -2 * x
    if y > 0:
      x = -3 * x
  else:
    y = 4 * x
  return x, y


def dependent_imbalanced_inner_if(x):
  y = 0
  if x > 0:
    if x < 3:
      y = -2 * x
      x = -3 * x
  return x, y


def independent_inner_for(a, b):
  p = 0
  for _ in a:
    tmp = b
    for j in tmp:
      p += j
  return p


def independent_inner_while(a, b):
  p = 0
  while a > 0:
    tmp = b
    while tmp > 0:
      p += 1
      tmp -= 1
    a -= 1
  return p


def dependent_inner_for(a, b):
  r = 1
  s = 0
  for _ in a:
    r += s
    tmp = b
    for j in tmp:
      s += j
  return r


def dependent_inner_while(a, b):
  r = 1
  while a > 0:
    r += 1
    tmp = b
    while tmp > 0:
      a -= 1
      tmp -= 1
  return r


def if_in_for(a):
  k = 0
  for i in a:
    if i % 2 > 0:
      j = i // 2
      k += j
  return k


def while_with_continue_in_context_manager(x):
  z = 0
  while x > 0:
    with tf.name_scope(''):
      x = x - 1
      if x < 5:
        continue
      z = z + 1
  return z


def while_continue_in_try(x):
  z = 0
  while x > 0:
    x = x - 1
    try:
      if x < 5:
        continue
      z = z + 1
    finally:
      z = z + 10
  return z


def while_break_in_context_manager(x):
  z = 0
  while x > 0:
    with tf.name_scope(''):
      x = x - 1
      if x < 5:
        break
      z = z + 1
  return z


def while_break_in_try(x):
  z = 0
  while x > 0:
    x = x - 1
    try:
      if x < 5:
        break
      z = z + 1
    finally:
      z = z + 10
  return z


class NestedControlFlowTest(reference_test_base.TestCase):

  def test_independent_ifs(self):
    self.assertNativeMatchesCompiled(independent_ifs, 1, 1)
    self.assertNativeMatchesCompiled(independent_ifs, 1, -1)
    self.assertNativeMatchesCompiled(independent_ifs, -1, 1)
    self.assertNativeMatchesCompiled(independent_ifs, -1, 1)

  def test_dependent_inner_if(self):
    self.assertNativeMatchesCompiled(dependent_inner_if, 1)
    self.assertNativeMatchesCompiled(dependent_inner_if, -1)

  def test_dependent_imbalanced_inner_if(self):
    self.assertNativeMatchesCompiled(dependent_imbalanced_inner_if, 1)
    self.assertNativeMatchesCompiled(dependent_imbalanced_inner_if, -1)

  def test_independent_inner_for(self):
    self.assertNativeMatchesCompiled(
        independent_inner_for, list(range(3)), list(range(5)))

  def test_independent_inner_while(self):
    self.assertNativeMatchesCompiled(independent_inner_while, 3, 5)

  def test_dependent_inner_for(self):
    self.assertNativeMatchesCompiled(
        dependent_inner_for, list(range(31)), list(range(7)))

  def test_dependent_inner_while(self):
    self.assertNativeMatchesCompiled(dependent_inner_while, 31, 7)

  def test_if_in_for(self):
    self.assertNativeMatchesCompiled(if_in_for, list(range(7)))

  def test_while_continue_in_context_manager(self):
    self.assertNativeMatchesCompiled(while_with_continue_in_context_manager, 10)
    self.assertNativeMatchesCompiled(while_with_continue_in_context_manager, 4)
    self.assertNativeMatchesCompiled(while_with_continue_in_context_manager, 0)

  def test_while_continue_in_try(self):
    self.assertNativeMatchesCompiled(while_continue_in_try, 10)
    self.assertNativeMatchesCompiled(while_continue_in_try, 4)
    self.assertNativeMatchesCompiled(while_continue_in_try, 0)

  def test_while_break_in_context_manager(self):
    self.assertNativeMatchesCompiled(while_break_in_context_manager, 10)
    self.assertNativeMatchesCompiled(while_break_in_context_manager, 4)
    self.assertNativeMatchesCompiled(while_break_in_context_manager, 0)

  def test_while_break_in_try(self):
    self.assertNativeMatchesCompiled(while_break_in_try, 10)
    self.assertNativeMatchesCompiled(while_break_in_try, 4)
    self.assertNativeMatchesCompiled(while_break_in_try, 0)

if __name__ == '__main__':
  tf.test.main()

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
"""Nested loops and loop control statements (e.g. break and continue).

Meant to verify that:
  * break/continue in the inner loop does not affect outer loop
  * break/continue inside nested conditionals still works
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reference_test_base
import tensorflow.compat.v1 as tf


def continue_in_single_for(l):
  s = 0
  for c in l:
    if c % 2 > 0:
      continue
    s += c
  return s


def continue_in_single_while(x):
  s = 0
  while x > 0:
    x -= 1
    if x % 2 > 0:
      continue
    s += x
  return s


def continue_in_inner_for(m):
  s = 0
  for l in m:
    for c in l:
      if c % 2 > 0:
        continue
      s += c
  return s


def continue_in_inner_while(x, y):
  s = 0
  while x > 0:
    x -= 1
    while y > 0:
      y -= 1
      if (x + y) % 2 > 0:
        continue
      s += x + y
  return s


def break_in_single_for(l):
  s = 0
  for c in l:
    if c % 2 > 0:
      break
    s += c
  return s


def break_in_single_while(x):
  s = 0
  while x > 0:
    x -= 1
    if x % 2 > 0:
      break
    s += x
  return s


def break_in_inner_for(m):
  s = 0
  for l in m:
    for c in l:
      if c % 2 > 0:
        break
      s += c
  return s


def break_in_inner_while(x, y):
  s = 0
  while x > 0:
    x -= 1
    while y > 0:
      y -= 1
      if ((x + y) % 2) == 0:
        break
      s += x + y
  return s


def break_continue_in_inner_for(m):
  s = 0
  for l in m:
    for c in l:
      if c % 2 > 0:
        break
      else:
        continue
      s += c
  return s


def break_continue_in_inner_while(x, y):
  s = 0
  while x > 0:
    x -= 1
    while y > 0:
      y -= 1
      if (x + y) % 2 > 0:
        break
      else:
        continue
      s += x + y
  return s


def break_followed_by_cond_in_single_for(x, y):
  for i in range(y):
    if i == 2:
      break
    if x > 0:
      x -= 1
  return x


def break_followed_by_cond_in_single_while(x):
  while x > 0:
    if x == 2:
      break
    if x > 0:
      x -= 1
  return x


def multiple_breaks_in_single_while(n):
  s = 1
  i = 0
  while i < n:
    i += 1
    if i > 10 * n:
      break
    if i == n:
      break
    s = s * 10 + i
  return i, s


class LoopControlFlowTest(reference_test_base.TestCase):

  def test_continue_in_single_for(self):
    self.assertNativeMatchesCompiled(continue_in_single_for,
                                     [1, 2, 3, 4, 5, 6])

  def test_continue_in_single_while(self):
    self.assertNativeMatchesCompiled(continue_in_single_while, 7)

  def test_continue_in_inner_for(self):
    self.assertNativeMatchesCompiled(continue_in_inner_for,
                                     [[1, 2, 3], [4, 5, 6]])

  def test_continue_in_inner_while(self):
    self.assertNativeMatchesCompiled(continue_in_inner_while, 10, 11)

  def test_break_in_single_for(self):
    self.assertNativeMatchesCompiled(break_in_single_for, [1, 2, 3, 4, 5, 6])

  def test_break_in_single_while(self):
    self.assertNativeMatchesCompiled(break_in_single_while, 7)

  def test_break_in_inner_for(self):
    self.assertNativeMatchesCompiled(break_in_inner_for,
                                     [[1, 2, 3], [4, 5, 6]])

  def test_break_in_inner_while(self):
    self.assertNativeMatchesCompiled(break_in_inner_while, 10, 11)

  def test_break_continue_in_inner_for(self):
    self.assertNativeMatchesCompiled(break_continue_in_inner_for,
                                     [[1, 2, 3], [4, 5, 6]])

  def test_break_continue_in_inner_while(self):
    self.assertNativeMatchesCompiled(break_continue_in_inner_while, 10, 11)

  def test_break_followed_by_cond_in_single_for(self):
    self.assertNativeMatchesCompiled(break_followed_by_cond_in_single_for, 3, 3)

  def test_break_followed_by_cond_in_single_while(self):
    self.assertNativeMatchesCompiled(break_followed_by_cond_in_single_while, 3)

  def test_multiple_breaks_in_single_while(self):
    self.assertNativeMatchesCompiled(multiple_breaks_in_single_while, 0)
    self.assertNativeMatchesCompiled(multiple_breaks_in_single_while, 2)
    self.assertNativeMatchesCompiled(multiple_breaks_in_single_while, 5)


if __name__ == '__main__':
  tf.test.main()

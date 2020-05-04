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
"""Reference tests check that a function is compiled correctly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import numbers
import os
import sys
import traceback

import numpy as np
import six
import tensorflow.compat.v2 as tf
import termcolor


class MutableContainer(object):
  """Testing helper that can create objects with properties."""

  def __init__(self, **kwargs):
    self.__dict__ = kwargs
    for k in kwargs:
      setattr(self, k, kwargs[k])

  def __str__(self):
    return 'MutableContainer%s' % self.__dict__

  def __ne__(self, other):
    return not self.__eq__(other)

  def __eq__(self, other):
    if not isinstance(other, MutableContainer):
      return False
    if self.__dict__.keys() != other.__dict__.keys():
      return False
    return all(
        self.__dict__[k] == other.__dict__[k] for k in self.__dict__.keys())


def to_graph(func, recursive=True):
  new_func = tf.autograph.to_graph(
      func,
      recursive=recursive,
      experimental_optional_features=tf.autograph.experimental.Feature.ALL)
  # TODO(b/127686409): Remove this.
  if inspect.ismethod(func):
    return six.create_bound_method(new_func, func.__self__)
  return new_func


def to_graph_nonrecursive(func):
  return to_graph(func, recursive=False)


def tf_function(func):
  return tf.function(func)


def tf_function_all(func):
  return tf.function(
      func,
      experimental_autograph_options=tf.autograph.experimental.Feature.ALL)


def tf_function_custom(options=None):
  def fn(func):
    return tf.function(
        func,
        experimental_autograph_options=options)
  return fn


class TestCase(tf.test.TestCase):
  """Base class for the reference tests."""

  def setUp(self):
    super(TestCase, self).setUp()

    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'
    # TODO(mdan): tf_function should be default here.
    self.convert = to_graph

  # TODO(mdan): Consider rewriting as a context manager.
  def _run_with_output_capture(self, func):
    out_capturer = six.StringIO()
    results = None
    captured_out = None
    captured_err = None
    try:
      sys.stdout = out_capturer
      results = func()
      captured_out = out_capturer.getvalue()
    except Exception as e:  # pylint:disable=broad-except
      sys.stdout = sys.__stdout__
      captured_err = e
      print('*** Capturing exception:\n{}\n'.format(traceback.format_exc()))
    finally:
      sys.stdout = sys.__stdout__
      out_capturer.close()
    return results, captured_out, captured_err

  def _as_tensors(self, args):
    tensor_args = []
    for a in args:
      if isinstance(a, (numbers.Number, list, np.ndarray)):
        tensor_arg = tf.constant(a)
      elif isinstance(a, dict):
        keys = tuple(a.keys())
        tensor_arg = dict(zip(keys, self._as_tensors([a[k] for k in keys])))
      elif isinstance(a, MutableContainer):
        tensor_arg = MutableContainer(**self._as_tensors([a.__dict__])[0])
      else:
        tensor_arg = a
      tensor_args.append(tensor_arg)
    return tensor_args

  def _as_ndarrays(self, args):
    return tuple(
        np.array(a) if isinstance(a, (numbers.Number, list, tuple)) else a
        for a in args
    )

  # TODO(mdan): Rename these to snake_case.
  def runCompiled(self, f, *args):
    return self.runTf(self.convert(f), *args)

  def runNumpy(self, f, *args):
    return self._run_with_output_capture(lambda: f(*self._as_ndarrays(args)))

  def runNative(self, f, *args):
    return self._run_with_output_capture(lambda: f(*args))

  def runTf(self, f, *args):
    with self.test_session() as sess:
      f_outs = f(*self._as_tensors(args))

      if isinstance(f_outs, tuple):
        outs = f_outs
      else:
        outs = (f_outs,)
      if f_outs is None:
        return None, '', None

      primitive_outs = tuple(
          o.__dict__ if isinstance(o, MutableContainer) else o for o in outs)
      # Convert any remaining primitives to tensors.
      primitive_outs = self._as_tensors(primitive_outs)

      (primitive_results, captured_out, captured_err
      ) = self._run_with_output_capture(lambda: sess.run(primitive_outs))
      if primitive_results is not None:
        final_outs = tuple(
            MutableContainer(**r) if isinstance(o, MutableContainer) else r
            for r, o in zip(primitive_results, outs))
      else:
        final_outs = (None,)

      if isinstance(f_outs, tuple):
        return final_outs, captured_out, captured_err
      else:
        return final_outs[0], captured_out, captured_err
      return final_outs

  def _deep_equal(self, left, right):
    if isinstance(left, tf.Tensor):
      return self._deep_equal(left.numpy(), right)
    if isinstance(right, tf.Tensor):
      return self._deep_equal(left, right.numpy())
    if isinstance(left, tf.SparseTensor) and isinstance(right, tf.SparseTensor):
      return (self._deep_equal(left.indices, right.indices)
              and self._deep_equal(left.values, right.values)
              and self._deep_equal(left.shape, right.shape))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
      return np.array_equal(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
      return all(self._deep_equal(l, r) for l, r in zip(left, right))
    return left == right

  def assertResultsMatch(self,
                         f,
                         args,
                         native_data,
                         compiled_data):
    native_results, native_out, native_err = native_data
    compiled_results, compiled_out, compiled_err = compiled_data
    str_args = '(%s)' % ', '.join(str(a) for a in args)
    # Using a manual verification to avoid a second compilation on success.
    # For exceptions, we don't enforce that they are the same, only that
    # both paths raised.
    # TODO(mdan): Add an API that returns both object and source code instead.
    outputs_equal = (
        self._deep_equal(native_results, compiled_results) and
        native_out == compiled_out)
    errors_equivalent = type(native_err) == type(compiled_err)  # pylint:disable=unidiomatic-typecheck
    if (not outputs_equal or not errors_equivalent):
      self.fail('Native and compiled functions are not equivalent.\n\n'
                'Native results: %s\n'
                'Compiled results: %s\n'
                'Native out: %s\n'
                'Compiled out: %s\n'
                'Native error: %s\n'
                'Compiled error: %s\n'
                'Native call: %s%s\n'
                'Check the logs for the generated code.'
                '' %
                (termcolor.colored(native_results, 'green', attrs=['bold']),
                 termcolor.colored(compiled_results, 'red', attrs=['bold']),
                 termcolor.colored(native_out, 'green', attrs=['bold']),
                 termcolor.colored(compiled_out, 'red', attrs=['bold']),
                 termcolor.colored(
                     '%s: %s' % (type(native_err).__name__, native_err),
                     'green',
                     attrs=['bold']),
                 termcolor.colored(
                     '%s: %s' % (type(compiled_err).__name__, compiled_err),
                     'red',
                     attrs=['bold']),
                 termcolor.colored(f.__name__, 'blue', attrs=['bold']),
                 termcolor.colored(str_args, 'blue', attrs=['bold'])))

  def assertFunctionMatchesEagerStatefulInput(self, f, args):
    """Like assertFunctionMatchesEager but creates new inputs each time."""
    compiled_data = self.runNative(tf.function(f), *args())
    native_data = self.runNative(f, *args())
    self.assertResultsMatch(f, args(), native_data, compiled_data)

  def assertFunctionMatchesEager(self, f, *args):
    compiled_data = self.runNative(tf.function(f), *args)
    native_data = self.runNative(f, *args)
    self.assertResultsMatch(f, args, native_data, compiled_data)

  def assertNativeMatchesCompiled(self, f, *args):
    compiled_data = self.runCompiled(f, *args)
    native_data = self.runNative(f, *args)
    self.assertResultsMatch(f, args, native_data, compiled_data)

  def assertTfMatchesCompiled(self, f, *args):
    compiled_data = self.runCompiled(f, *args)
    native_data = self.runTf(f, *args)
    self.assertResultsMatch(f, args, native_data, compiled_data)

  def assertNativeMatchesCompiledMethod(self, m, *args):
    compiled_data = self.runCompiled(m, *args)
    native_data = self.runNative(m, *args)
    self.assertResultsMatch(m, args, native_data, compiled_data)

  def assertMatchesObject(self, c, methods_and_args, native_run, compiled_run):
    init_func, init_args = methods_and_args[0]
    assert init_func == '__init__'

    native_object = c(*init_args)
    compiled_c = self.convert(c)
    compiled_object = compiled_c(*self._as_tensors(init_args))
    for name, args in methods_and_args[1:]:
      native_method = getattr(native_object, name)
      compiled_method = getattr(compiled_object, name)
      native_data = native_run(native_method, *args)
      compiled_data = compiled_run(compiled_method, *args)
      self.assertResultsMatch(native_method, args, native_data, compiled_data)

  def assertTfMatchesCompiledObject(self, c, methods_and_args):
    self.assertMatchesObject(c, methods_and_args, self.runTf, self.runTf)

  def assertNativeMatchesCompiledObject(self, c, methods_and_args):
    self.assertMatchesObject(c, methods_and_args, self.runNative, self.runTf)

  def try_execute_compiled(self, f, *args):
    _, _, err = self.runCompiled(f, *args)
    if err:
      raise err


if __name__ == '__main__':
  tf.test.main()

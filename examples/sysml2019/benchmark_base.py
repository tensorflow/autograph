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
"""Common benchmarking code.

See https://www.tensorflow.org/community/benchmarks for usage.
To run all benchmarks, use "--benchmarks=.".
Control the output directory using the "TEST_REPORT_FILE_PREFIX" environment
variable.

For the benchmarks in this directory, we used:

    TEST_REPORT_FILE_PREFIX=/tmp/autograph/sysml2019_benchmarks/

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf


class ReportingBenchmark(tf.test.Benchmark):
  """Base class for a benchmark that reports general performance metrics."""

  def time_execution(self,
                     name,
                     target,
                     iters=None,
                     warm_up_iters=3,
                     iter_volume=None,
                     iter_unit=None,
                     extras=None):
    if iters is None:
      iters = int(os.environ.get('BENCHMARK_NUM_EXECUTIONS', 50))

    for _ in range(warm_up_iters):
      target()

    all_times = []
    for _ in range(iters):
      iter_time = time.time()
      target()
      all_times.append(time.time() - iter_time)

    extras = dict(extras) if extras else {}

    extras['all_times'] = all_times

    extras['name'] = name
    if isinstance(name, tuple):
      name = '_'.join(str(piece) for piece in name)

    # TODO(mdanatg): This is unnecessary - use normal extras.
    if iter_volume is not None:
      assert iter_unit is not None
      extras['iter_volume'] = iter_volume
      extras['iter_unit'] = iter_unit

    self.report_benchmark(
        iters=iters, wall_time=sum(all_times), name=name, extras=extras)


if __name__ == '__main__':
  tf.test.main()

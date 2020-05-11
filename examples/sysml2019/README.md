To run the benchmarks on Linux, use the commands below.

Tip: use a virtual environment to avoid clobbering your existing TF installation.

Create a directory for the benchmark result files:

    export BENCHMARKS_DIR=/tmp/autograph/sysml2019_benchmarks/
    export TEST_REPORT_FILE_PREFIX=${BENCHMARKS_DIR}
    export BENCHMARK_NUM_EXECUTIONS=100
    # Optionally, clean up the target directory: rm -Rf ${BENCHMARKS_DIR}
    mkdir -p ${BENCHMARKS_DIR}

Run benchmarks:

    pip install tensorflow
    python beam_search_benchmark.py --benchmarks=.
    python lbfgs_benchmark.py --benchmarks=.
    python maml_benchmark.py --benchmarks=.
    python seq2seq_benchmark.py --benchmarks=.

To parse the result files, start a Jupyter client and use the notebook:

    pip install jupyter
    pip install pandas
    jupyter notebook benchmark_dashboard.ipynb

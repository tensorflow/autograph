# Autograph Internals

This directory contains tests for Python idioms that AutoGraph supports.
Since they are easy to read, they also double of small code samples.

The BUILD file contains the full list of tests.

## Locating the samples inside tests

Each test is structured as:

    <imports>

    <sample functions>

    <test class>

The sample functions are what demonstrate how code is authored for AutoGraph.

The test in generale ensure that the sample code produces the same results when
run in a TF graph as it would when executed as regular Python.

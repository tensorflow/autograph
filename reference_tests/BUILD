licenses(["notice"])

exports_files(["LICENSE"])

load(":reference_test.bzl", "reference_test")

py_library(
    name = "reference_tests",
    srcs = ["reference_test_base.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
        "//third_party/py/termcolor",
    ],
)

# # Misc

# ## Boolean expressions, equality testing

reference_test(
    name = "logical_expression_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

# # Scoping and modularity

# ## Composite names

reference_test(
    name = "composite_names_in_control_flow_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

# ## Function Calls

reference_test(
    name = "call_to_cast_function_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "call_to_builtin_function_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "call_to_lambda_function_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "call_to_named_tuple_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "call_to_print_function_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "call_to_tf_function_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "two_level_call_tree_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "dynamic_call_tree_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

# # Control Flow

reference_test(
    name = "basic_ifexp_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "cond_basic_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "early_return_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "loop_basic_test",
    shard_count = 10,
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "loop_control_flow_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(name = "loop_scoping_test")

reference_test(name = "loop_with_else_test")

reference_test(
    name = "loop_with_variable_type_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "loop_with_variable_type_illegal_cases_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "loop_with_function_call_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "nested_control_flow_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "while_loop_function_call_mix_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

# ## Assert

reference_test(
    name = "assertion_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

# # Data Structures

# ## Generators

reference_test(name = "generator_test")

# ## Iterators

reference_test(
    name = "datasets_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

reference_test(
    name = "distributed_dataset_test",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

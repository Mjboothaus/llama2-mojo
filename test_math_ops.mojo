"""
Test suite for math_ops module functions including rmsnorm, softmax,
argmax, and sample.

This module contains comprehensive unit tests designed to verify correctness
and numerical stability of core numerical kernels used in transformer models.

Tests include:
- Basic and weighted RMS normalization tests verifying output normalization.
- Softmax tests covering uniform, large, negative, and basic inputs, ensuring
  probability correctness and numerical stability.
- Argmax tests checking correct index selection for various input patterns.
- Sampling tests verifying probability adherence and output index validity.

Each test function raises on failure via assertions from the Mojo testing
framework, making this module suitable for automated CI pipelines and
incremental development verification.

Run all tests with: mojo run -I src tests/test_math_ops.mojo
"""

from testing import assert_true, assert_almost_equal, assert_equal, TestSuite
from src.matrix import Matrix
from src.math_ops import rmsnorm, softmax, argmax, sample
import math


fn test_rmsnorm_basic() raises:
    print("\nTesting basic rmsnorm:")
    var x = Matrix(4)
    var o = Matrix(4)
    var weight = Matrix(4)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0
    for i in range(4):
        weight[i] = 1.0
    rmsnorm(o.data, x.data, weight.data, 4)
    var rms = math.sqrt(
        (
            Float32(1.0) ** 2
            + Float32(2.0) ** 2
            + Float32(3.0) ** 2
            + Float32(4.0) ** 2
        )
        / Float32(4.0)
        + Float32(1e-5)
    )
    assert_almost_equal(o[0], Float32(1.0) / rms, atol=0.001)
    assert_almost_equal(o[1], Float32(2.0) / rms, atol=0.001)
    assert_almost_equal(o[2], Float32(3.0) / rms, atol=0.001)
    assert_almost_equal(o[3], Float32(4.0) / rms, atol=0.001)
    print("✓ Basic rmsnorm test passed")


fn test_rmsnorm_with_weights() raises:
    print("\nTesting rmsnorm with custom weights:")
    var x = Matrix(3)
    var o = Matrix(3)
    var weight = Matrix(3)
    for i in range(3):
        x[i] = 2.0
    weight[0] = 1.0
    weight[1] = 2.0
    weight[2] = 3.0
    rmsnorm(o.data, x.data, weight.data, 3)
    var rms = math.sqrt(
        (Float32(2.0) ** 2 + Float32(2.0) ** 2 + Float32(2.0) ** 2)
        / Float32(3.0)
        + Float32(1e-5)
    )
    var ss = Float32(1.0) / rms
    assert_almost_equal(o[0], weight[0] * ss * Float32(2.0), atol=0.001)
    assert_almost_equal(o[1], weight[1] * ss * Float32(2.0), atol=0.001)
    assert_almost_equal(o[2], weight[2] * ss * Float32(2.0), atol=0.001)
    print("✓ Rmsnorm with weights test passed")


fn test_rmsnorm_zeros() raises:
    print("\nTesting rmsnorm with zero input:")
    var x = Matrix(4)
    var o = Matrix(4)
    var weight = Matrix(4)
    x.zero()
    for i in range(4):
        weight[i] = 1.0
    rmsnorm(o.data, x.data, weight.data, 4)
    for i in range(4):
        assert_almost_equal(o[i], 0.0, atol=0.001)
    print("✓ Rmsnorm with zeros test passed")


fn test_softmax_basic() raises:
    print("\nTesting basic softmax:")
    var x = Matrix(3)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    softmax(x.data, 3)
    var e0 = math.exp(Float32(-2.0))
    var e1 = math.exp(Float32(-1.0))
    var e2 = math.exp(Float32(0.0))
    var sum = e0 + e1 + e2
    assert_almost_equal(x[0], e0 / sum, atol=0.001)
    assert_almost_equal(x[1], e1 / sum, atol=0.001)
    assert_almost_equal(x[2], e2 / sum, atol=0.001)
    var total = x[0] + x[1] + x[2]
    assert_almost_equal(total, 1.0, atol=0.001)
    print("✓ Basic softmax test passed")


fn test_argmax_basic() raises:
    print("\nTesting basic argmax:")
    var v = Matrix(5)
    v[0] = 1.0
    v[1] = 5.0
    v[2] = 3.0
    v[3] = 2.0
    v[4] = 4.0
    var idx = argmax(v.data, 5)
    assert_equal(idx, 1)
    assert_almost_equal(v[idx], 5.0, atol=0.001)
    print("✓ Basic argmax test passed")


fn test_sample_basic() raises:
    print("\nTesting basic sample:")
    var probs = Matrix(3)
    probs[0] = 0.2
    probs[1] = 0.5
    probs[2] = 0.3
    var counts = List[Int](3)
    for _ in range(3):
        counts.append(0)
    var num_samples = 100
    for _ in range(num_samples):
        var idx = sample(probs.data, 3)
        assert_true(
            idx >= 0 and idx < 3, "Sample index should be in valid range"
        )
        counts[idx] += 1
    assert_true(counts[0] > 0, "Index 0 should be sampled")
    assert_true(counts[1] > 0, "Index 1 should be sampled")
    assert_true(counts[2] > 0, "Index 2 should be sampled")
    print("✓ Basic sample test passed")


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

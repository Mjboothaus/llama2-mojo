from src.config import Config
from testing import assert_equal, assert_true, TestSuite

# Since Config constructor reads from file, we need mock file paths for testing
# Assume you have appropriate small config files available for testing


# Example test to verify Config fields from a known config file
fn test_config_initialization() raises:
    var config = Config("testdata/config.bin", False)
    assert_true(config.dim > 0)
    assert_true(config.hidden_dim > 0)
    assert_true(config.n_layers > 0)
    assert_true(config.n_heads > 0)
    assert_true(config.n_kv_heads > 0)
    assert_true(config.vocab_size != 0)
    assert_true(config.seq_len > 0)
    assert_true(config.head_size == Int(config.dim / config.n_heads))
    assert_true(
        config.kv_dim == Int((config.n_kv_heads * config.dim) / config.n_heads)
    )
    assert_true(config.kv_mul == Int(config.n_heads / config.n_kv_heads))
    assert_true(
        (config.shared_weights and config.vocab_size > 0)
        or (not config.shared_weights and config.vocab_size < 0)
    )


# Example test with print enabled (output captured by test runner)
fn test_config_print_output() raises:
    var config = Config("testdata/config.bin", True)
    assert_true(config.dim > 0)  # basic check to proceed


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

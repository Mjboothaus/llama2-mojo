from src.utils import str_concat, string_compare, wrap, string_from_bytes
from testing import assert_equal, assert_true, TestSuite


fn test_str_concat() raises:
    assert_equal(str_concat("hello", "world"), "helloworld")
    assert_equal(str_concat("", ""), "")
    assert_equal(str_concat("foo", ""), "foo")
    assert_equal(str_concat("", "bar"), "bar")


fn test_string_compare() raises:
    assert_equal(string_compare("a", "b"), -1)
    assert_equal(string_compare("b", "a"), 1)
    assert_equal(string_compare("same", "same"), 0)


fn test_wrap() raises:
    assert_equal(wrap("\\n"), "\n")
    assert_equal(wrap("\\t"), "\t")
    assert_equal(wrap("'"), "'")
    assert_equal(wrap('"'), '"')

    # unchanged if no match
    assert_equal(wrap("x"), "x")


fn test_string_from_bytes() raises:
    var bytes = [
        UInt8(72),
        UInt8(101),
        UInt8(108),
        UInt8(108),
        UInt8(111),
    ]  # "Hello"
    assert_equal(string_from_bytes(bytes.copy()), "Hello")

    var empty: List[UInt8] = []
    assert_equal(string_from_bytes(empty.copy()), "")


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

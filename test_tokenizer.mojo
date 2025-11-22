from testing import assert_true, assert_equal
from src.tokenizer import Tokenizer


fn test_tokenizer() raises:
    print("\nTesting Tokenizer loading:")
    var tok = Tokenizer(32000, "tokenizer.bin")

    print("  Tokenizer loaded:")
    print("    vocab_size:", tok.vocab_size)
    print("    max_token_length:", tok.max_token_length)
    print("    vocab length:", len(tok.vocab))
    print("    vocab_scores length:", len(tok.vocab_scores))

    assert_equal(tok.vocab_size, 32000)
    assert_equal(tok.max_token_length, 27)
    assert_equal(len(tok.vocab), 32000)
    assert_equal(len(tok.vocab_scores), 32000)

    assert_true(len(tok.vocab[0]) >= 0, "Token 0 should exist")
    assert_true(len(tok.vocab[1]) >= 0, "Token 1 (BOS) should exist")
    assert_true(len(tok.vocab[2]) >= 0, "Token 2 (EOS) should exist")

    print("\n  Sample tokens:")
    for i in range(min(10, tok.vocab_size)):
        print(
            "    Token",
            i,
            ":",
            repr(tok.vocab[i]),
            "score:",
            tok.vocab_scores[i],
        )

    print("✓ Tokenizer loading test passed!")


fn test_tokenizer_find() raises:
    print("\nTesting Tokenizer.find() method:")
    var tok = Tokenizer(32000, "tokenizer.bin")

    var token_5 = tok.vocab[5]
    var token_10 = tok.vocab[10]
    var token_100 = tok.vocab[100]

    var idx_5 = tok.find(token_5)
    var idx_10 = tok.find(token_10)
    var idx_100 = tok.find(token_100)

    print("  Finding token at index 5:", repr(token_5), "-> found at:", idx_5)
    print(
        "  Finding token at index 10:", repr(token_10), "-> found at:", idx_10
    )
    print(
        "  Finding token at index 100:",
        repr(token_100),
        "-> found at:",
        idx_100,
    )

    assert_true(
        idx_5 >= 0 and idx_5 < tok.vocab_size, "Valid index for token 5"
    )
    assert_true(
        idx_10 >= 0 and idx_10 < tok.vocab_size, "Valid index for token 10"
    )
    assert_true(
        idx_100 >= 0 and idx_100 < tok.vocab_size, "Valid index for token 100"
    )

    assert_equal(
        tok.vocab[idx_5], token_5, "Found token matches original token 5"
    )
    assert_equal(
        tok.vocab[idx_10], token_10, "Found token matches original token 10"
    )
    assert_equal(
        tok.vocab[idx_100], token_100, "Found token matches original token 100"
    )

    var idx_1 = tok.find(tok.vocab[1])
    var idx_2 = tok.find(tok.vocab[2])
    assert_equal(idx_1, 1, "Find BOS token at index 1")
    assert_equal(idx_2, 2, "Find EOS token at index 2")

    var not_found = tok.find("NON_EXISTENT_TOKEN_123456")
    print("  Finding non-existent token -> found at:", not_found)
    assert_equal(not_found, -1, "Returns -1 for non-existent token")

    print("✓ Tokenizer.find() test passed!")


fn main() raises:
    print("=" * 60)
    print("Testing Tokenizer")
    print("=" * 60)
    print()

    test_tokenizer()
    test_tokenizer_find()

    print("\nAll tokenizer tests passed! ✓")

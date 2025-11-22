""" 
Tokenizer module

Implements a simple vocabulary tokenizer supporting binary
vocab formats, token lookup, and token merging. Designed for
fast loading and tokenization of model inputs.
"""

from src.utils import wrap, string_from_bytes, str_concat

alias element_type = Float32


struct Tokenizer:
    var vocab: List[String]
    var vocab_scores: List[element_type]
    var max_token_length: Int
    var vocab_size: Int
    var map_vocab_to_index: Dict[String, Int]

    fn __init__(out self, vocab_size: Int, filename: String) raises:
        with open(filename, "r") as f:

            @parameter
            fn read_bytes_as[dtype: DType](size: Int) raises -> SIMD[dtype, 1]:
                var bytes = f.read_bytes(size)
                var result = bytes.unsafe_ptr().bitcast[SIMD[dtype, 1]]()[0]
                return result

            self.vocab_size = vocab_size
            self.vocab_scores = List[element_type]()
            self.vocab = List[String]()

            var max_token_bytes = f.read_bytes(4)
            var max_token_ptr = max_token_bytes.unsafe_ptr().bitcast[Int32]()
            self.max_token_length = Int(max_token_ptr[0])

            self.map_vocab_to_index = Dict[String, Int]()

            for i in range(self.vocab_size):
                var score = read_bytes_as[DType.float32](4)
                var slen = read_bytes_as[DType.int32](4)
                var token = string_from_bytes(f.read_bytes(Int(slen)))
                self.vocab.append(token)
                self.vocab_scores.append(score)
                self.map_vocab_to_index[self.vocab[i]] = i

    fn find(self, token_o: String) raises -> Int:
        var token = wrap(token_o)
        var index = self.map_vocab_to_index.find(token)
        return index.or_else(-1)

    fn print_tokens(self, n: Int):
        var count = min(n, self.vocab_size)
        print("First", count, "tokens:")
        for i in range(count):
            print(i, ":", self.vocab[i])


fn bpe_encode(mut tokens: List[Int], text: String, tok: Tokenizer) raises:
    for pos in range(len(text)):
        var char = String(text[pos])
        var id = tok.find(char)
        if id == -1:
            print("Not a good prompt token at pos ", pos)
            return
        tokens.append(id)

    while True:
        var best_score = Float32(-1e10)
        var best_id = -1
        var best_idx = -1

        for i in range(len(tokens) - 1):
            var str = str_concat(tok.vocab[tokens[i]], tok.vocab[tokens[i + 1]])
            var id = tok.find(str)
            if id != -1 and tok.vocab_scores[id] > best_score:
                best_score = tok.vocab_scores[id]
                best_id = id
                best_idx = i

        if best_idx == -1:
            break

        tokens[best_idx] = best_id
        var _tokens = List[Int]()
        for i in range(0, best_idx + 1):
            _tokens.append(tokens[i])
        for i in range(best_idx + 2, len(tokens)):
            _tokens.append(tokens[i])
        tokens = _tokens^

""" 
Config module

Defines the model configuration structure and methods to parse
config parameters from checkpoint files for transformer models.
"""

alias element_type = DType.float32
alias NUM_CONFIG_INT = 7


struct Config:
    var dim: Int
    var kv_dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var kv_mul: Int
    var vocab_size: Int
    var seq_len: Int
    var head_size: Int
    var shared_weights: Bool

    fn __init__(out self, filename: String, print_config: Bool) raises:
        var f = open(filename, "r")
        var bytes_of_config_params = NUM_CONFIG_INT * size_of[DType.int32]()
        var config_data_raw = f.read_bytes(bytes_of_config_params)
        f.close()
        var int32_ptr = config_data_raw.steal_data().bitcast[Int32]()
        self.dim = Int(int32_ptr[0])
        self.hidden_dim = Int(int32_ptr[1])
        self.n_layers = Int(int32_ptr[2])
        self.n_heads = Int(int32_ptr[3])
        self.n_kv_heads = Int(int32_ptr[4])
        self.vocab_size = Int(int32_ptr[5])
        self.seq_len = Int(int32_ptr[6])
        self.head_size = self.dim // self.n_heads
        self.kv_dim = (self.n_kv_heads * self.dim) // self.n_heads
        self.kv_mul = self.n_heads // self.n_kv_heads
        self.shared_weights = self.vocab_size > 0
        if not self.shared_weights:
            self.vocab_size = -self.vocab_size

        if print_config:
            print("config: dim, hidden_dim", self.dim, self.hidden_dim)
            print("config: n_layers, n_heads", self.n_layers, self.n_heads)
            print("config: vocab_size, seq_len", self.vocab_size, self.seq_len)
            print("config: head_size", self.head_size)
            print("config: kv_dim, kv_mul", self.kv_dim, self.kv_mul)

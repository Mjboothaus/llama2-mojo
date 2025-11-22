/// Weights module
///
/// Provides data structures and loading logic for transformer model
/// parameters including embedding tables, attention matrices, and
/// feedforward weights.


from matrix import Matrix
from config import Config

alias element_type = DType.float32
alias NUM_CONFIG_INT = 7

struct TransformerWeights:
    var token_embedding_table: Matrix
    var freq_cis_real: Matrix
    var freq_cis_imag: Matrix
    var rms_att_weight: Matrix
    var wq: Matrix
    var wk: Matrix
    var wv: Matrix
    var wo: Matrix
    var rms_ffn_weight: Matrix
    var w1: Matrix
    var w3: Matrix
    var w2: Matrix
    var rms_final_weight: Matrix
    var wcls: Matrix

    fn __init__(out self, file_name: String, config: Config) raises:
        var bytes_read = 0
        var f = open(file_name, "r")

        _ = f.read_bytes(NUM_CONFIG_INT * size_of[DType.int32]())
        bytes_read += NUM_CONFIG_INT * size_of[DType.int32]()

        @parameter
        fn read_weights(*dims: Int) raises -> Matrix:
            var dim_list = List[Int]()
            var num_elements = 1
            for i in range(len(dims)):
                dim_list.append(dims[i])
                num_elements *= dims[i]

            var tmp = f.read_bytes(num_elements * size_of[element_type]())
            bytes_read += num_elements * size_of[element_type]()
            var data = tmp.steal_data().bitcast[element_type]()
            return Matrix(data, dim_list^)

        self.token_embedding_table = read_weights(config.vocab_size, config.dim)
        self.rms_att_weight = read_weights(config.n_layers, config.dim)
        self.wq = read_weights(config.n_layers, config.dim, config.dim)
        self.wk = read_weights(config.n_layers, config.kv_dim, config.dim)
        self.wv = read_weights(config.n_layers, config.kv_dim, config.dim)
        self.wo = read_weights(config.n_layers, config.dim, config.dim)
        self.rms_ffn_weight = read_weights(config.n_layers, config.dim)
        self.w1 = read_weights(config.n_layers, config.hidden_dim, config.dim)
        self.w2 = read_weights(config.n_layers, config.dim, config.hidden_dim)
        self.w3 = read_weights(config.n_layers, config.hidden_dim, config.dim)
        self.rms_final_weight = read_weights(config.dim)
        self.freq_cis_real = read_weights(config.seq_len, config.head_size // 2)
        self.freq_cis_imag = read_weights(config.seq_len, config.head_size // 2)

        if config.shared_weights:
            var dims = self.token_embedding_table.dims.copy()
            self.wcls = Matrix(self.token_embedding_table.data, dims^)
            self.wcls.allocated = 0
        else:
            self.wcls = read_weights(config.vocab_size, config.dim)

        f.close()

        print(
            "Total bytes read:",
            bytes_read,
            "Estimated checkpoint size: ",
            bytes_read // 1024 // 1024,
            "MB",
        )

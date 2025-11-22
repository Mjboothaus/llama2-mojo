/// Transformer module
///
/// Implements the core transformer block including attention mechanisms,
/// rotary positional embeddings (RoPE), feedforward networks,
/// residual connections, and the transformer forward pass.


from matrix import Matrix
from config import Config
from weights import TransformerWeights
from math_ops import rmsnorm, softmax, matmul, batch_matmul, add
from sys.info import simd_width_of, num_performance_cores

alias element_type = DType.float32
alias n_element = (4 * simd_width_of[element_type]())

struct RunState:
    var x: Matrix
    var xb: Matrix
    var xb2: Matrix
    var hb: Matrix
    var hb2: Matrix
    var q: Matrix
    var att: Matrix
    var logits: Matrix
    var key_cache: Matrix
    var value_cache: Matrix

    fn __init__(out self, config: Config) raises:
        self.x = Matrix(config.dim)
        self.xb = Matrix(config.dim)
        self.xb2 = Matrix(config.dim)
        self.hb = Matrix(config.hidden_dim)
        self.hb2 = Matrix(config.hidden_dim)
        self.q = Matrix(config.dim)
        self.logits = Matrix(config.vocab_size)
        self.att = Matrix(config.n_heads, config.seq_len)
        self.key_cache = Matrix(config.n_layers, config.seq_len, config.kv_dim)
        self.value_cache = Matrix(config.n_layers, config.seq_len, config.kv_dim)

struct Transformer:
    var workers: Int

    fn __init__(out self, workers: Int = num_performance_cores()):
        self.workers = workers

    @always_inline
    fn rope_rotation_llama(
        self,
        q_ptr: UnsafePointer[element_type, MutOrigin.external],
        k_ptr: UnsafePointer[element_type, MutOrigin.external],
        freq_cis_real_row: UnsafePointer[element_type, MutOrigin.external],
        freq_cis_imag_row: UnsafePointer[element_type, MutOrigin.external],
        config: Config,
        head_size: Int
    ):
        @parameter
        fn head_loop(i: Int):
            for j in range(0, head_size, 2):
                var fcr = freq_cis_real_row[j // 2]
                var fci = freq_cis_imag_row[j // 2]

                var q_idx = i * head_size + j
                var q0 = q_ptr[q_idx]
                var q1 = q_ptr[q_idx + 1]
                q_ptr[q_idx] = q0 * fcr - q1 * fci
                q_ptr[q_idx + 1] = q0 * fci + q1 * fcr

                if i < config.n_kv_heads:
                    var k_idx = i * head_size + j
                    var k0 = k_ptr[k_idx]
                    var k1 = k_ptr[k_idx + 1]
                    k_ptr[k_idx] = k0 * fcr - k1 * fci
                    k_ptr[k_idx + 1] = k0 * fci + k1 * fcr

        parallelize[head_loop](config.n_heads, self.workers)

    @always_inline
    fn transformer(
        self,
        token: Int,
        pos: Int,
        config: Config,
        mut state: RunState,
        weights: TransformerWeights,
    ) raises:
        var dim = config.dim
        var hidden_dim = config.hidden_dim
        var head_size = config.head_size
        var kv_dim = config.kv_dim
        var kv_mul = config.kv_mul
        var sqrt_head_size = math.sqrt[dtype=DType.float32, width=1](element_type(head_size))

        var content_row = weights.token_embedding_table.slice(token)
        memcpy(dest=state.x.data, src=content_row, count=dim)

        var freq_cis_real_row = weights.freq_cis_real.slice(pos)
        var freq_cis_imag_row = weights.freq_cis_imag.slice(pos)

        for l in range(config.n_layers):
            rmsnorm(state.xb.data, state.x.data, weights.rms_att_weight.slice(l), dim)

            var loff = l * config.seq_len * config.kv_dim
            var k_ptr = state.key_cache.slice(l, pos)
            var v_ptr = state.value_cache.slice(l, pos)

            if kv_dim == dim:
                batch_matmul[3](
                    StaticTuple[UnsafePointer[element_type, MutOrigin.external], 3](
                        state.q.data, k_ptr, v_ptr
                    ),
                    state.xb.data,
                    StaticTuple[UnsafePointer[element_type, MutOrigin.external], 3](
                        weights.wq.slice(l),
                        weights.wk.slice(l),
                        weights.wv.slice(l),
                    ),
                    dim,
                    dim,
                    self.workers,
                )
            else:
                matmul(state.q.data, state.xb.data, weights.wq.slice(l), dim, dim, self.workers)
                batch_matmul[2](
                    StaticTuple[UnsafePointer[element_type, MutOrigin.external], 2](
                        k_ptr, v_ptr
                    ),
                    state.xb.data,
                    StaticTuple[UnsafePointer[element_type, MutOrigin.external], 2](
                        weights.wk.slice(l),
                        weights.wv.slice(l),
                    ),
                    kv_dim,
                    dim,
                    self.workers,
                )

            self.rope_rotation_llama(state.q.data, k_ptr, freq_cis_real_row, freq_cis_imag_row, config, head_size)
            memset_zero(state.xb.data, state.xb.size())

            @parameter
            fn loop_over_heads(h: Int):
                var q_offset = h * head_size
                var att_offset = h * config.seq_len

                for t in range(pos + 1):
                    var k_offset = loff + t * kv_dim + (h // kv_mul) * head_size
                    var score: element_type = 0.0

                    @parameter
                    fn score_fn[_nelts: Int](i: Int):
                        score += (
                            state.q.data.load[width=_nelts](q_offset + i)
                                * state.key_cache.data.load[width=_nelts](k_offset + i)
                        ).reduce_add()

                    vectorize[score_fn, n_element](head_size)
                    score /= sqrt_head_size
                    state.att.data[att_offset + t] = score

                softmax(state.att.data, att_offset, att_offset + pos + 1)

                var xb_offset = h * head_size
                for t in range(pos + 1):
                    var v_offset = loff + t * kv_dim + (h // kv_mul) * head_size
                    var a = state.att.data[att_offset + t]

                    @parameter
                    fn xb_accumulate[_nelts: Int](i: Int):
                        var xbi = state.xb.data.offset(xb_offset + i).load[width=_nelts](0) + a * state.value_cache.data.offset(v_offset + i).load[width=_nelts](0)
                        state.xb.data.offset(xb_offset + i).store[width=_nelts](0, xbi)

                    vectorize[xb_accumulate, n_element](head_size)

            parallelize[loop_over_heads](config.n_heads, self.workers)

            matmul(state.xb2.data, state.xb.data, weights.wo.slice(l), dim, dim, self.workers)
            add(state.x.data, state.xb2.data, dim)
            rmsnorm(state.xb.data, state.x.data, weights.rms_ffn_weight.slice(l), dim)

            batch_matmul[2](
                StaticTuple[UnsafePointer[element_type, MutOrigin.external], 2](state.hb.data, state.hb2.data),
                state.xb.data,
                StaticTuple[UnsafePointer[element_type, MutOrigin.external], 2](
                    weights.w1.slice(l),
                    weights.w3.slice(l),
                ),
                hidden_dim,
                dim,
                self.workers,
            )

            @parameter
            fn silu[_nelts: Int](i: Int):
                var initial_hb = state.hb.data.offset(i).load[width=_nelts](0)
                var hbi = initial_hb * (1.0 / (1.0 + math.exp(-initial_hb)))
                state.hb.data.offset(i).store[width=_nelts](0, hbi * state.hb2.data.offset(i).load[width=_nelts](0))

            vectorize[silu, n_element](hidden_dim)
            matmul(state.xb.data, state.hb.data, weights.w2.slice(l), dim, hidden_dim, self.workers)
            add(state.x.data, state.xb.data, dim)

        rmsnorm(state.x.data, state.x.data, weights.rms_final_weight.data, dim)
        matmul(state.logits.data, state.x.data, weights.wcls.data, config.vocab_size, dim, self.workers)

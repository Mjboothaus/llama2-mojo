/// Mathematical operations module
///
/// Implements essential neural network math kernels such as RMSNorm,
/// softmax, matrix multiplication with SIMD vectorization and
/// parallelization for CPU backends.


alias element_type = DType.float32
alias n_element = (4 * simd_width_of[element_type]())

@always_inline
fn rmsnorm(
    mut o: UnsafePointer[element_type, MutOrigin.external],
    x: UnsafePointer[element_type, MutOrigin.external],
    weight: UnsafePointer[element_type, MutOrigin.external],
    size: Int
):
    var tmp_ptr = stack_allocation[n_element, element_type]()
    tmp_ptr.store[width=n_element](0, SIMD[element_type, n_element](0))

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        var val = x.offset(j).load[width=_nelts](0) ** 2
        var curr = tmp_ptr.load[width=_nelts](0)
        tmp_ptr.store[width=_nelts](0, curr + val)

    vectorize[_sum2, n_element](size)

    var ss: element_type = tmp_ptr.load[width=n_element](0).reduce_add()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    @parameter
    fn _norm[_nelts: Int](j: Int):
        var val = weight.load[width=_nelts](j) * ss * x.load[width=_nelts](j)
        o.offset(j).store[width=_nelts](0, val)

    vectorize[_norm, n_element](size)

@always_inline
fn softmax(mut x: UnsafePointer[element_type, MutOrigin.external], size: Int):
    softmax(x, 0, size)

@always_inline
fn softmax(mut x: UnsafePointer[element_type, MutOrigin.external], start: Int, end: Int):
    var max_val: element_type = -1e9

    @parameter
    fn _max[_nelts: Int](ii: Int):
        var val = x.load[width=_nelts](start + ii).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[_max, n_element](end - start)

    var acc_ptr = stack_allocation[n_element, element_type]()
    acc_ptr.store[width=n_element](0, SIMD[element_type, n_element](0))

    @parameter
    fn _exp[_nelts: Int](ii: Int):
        var val = math.exp(x.load[width=_nelts](start + ii) - max_val)
        x.store[width=_nelts](start + ii, val)
        var curr = acc_ptr.load[width=_nelts](0)
        acc_ptr.store[width=_nelts](0, curr + val)

    vectorize[_exp, n_element](end - start)

    var ssum = acc_ptr.load[width=n_element](0).reduce_add()

    @parameter
    fn _norm[_nelts: Int](ii: Int):
        x.store[width=_nelts](start + ii, x.load[width=_nelts](start + ii) / ssum)

    vectorize[_norm, n_element](end - start)

@always_inline
fn batch_matmul[
    n: Int
](
    C: StaticTuple[UnsafePointer[element_type, MutOrigin.external], n],
    A: UnsafePointer[element_type, MutOrigin.external],
    B: StaticTuple[UnsafePointer[element_type, MutOrigin.external], n],
    rows: Int,
    cols: Int,
    workers: Int,
):
    @parameter
    fn compute_row(i: Int):
        var tmp_ptr = stack_allocation[n * n_element, element_type]()
        @parameter
        for k in range(n):
            tmp_ptr.store[width=n_element](k * n_element, SIMD[element_type, n_element](0))
        var row_offset = i * cols
        @parameter
        fn dot[_nelts: Int](j: Int):
            var a = A.offset(j).load[width=_nelts](0)
            @parameter
            for k in range(n):
                var val = a * B[k].offset(row_offset + j).load[width=_nelts](0)
                var curr = tmp_ptr.load[width=_nelts](k * n_element)
                tmp_ptr.store[width=_nelts](k * n_element, curr + val)
        vectorize[dot, n_element](cols)
        @parameter
        for k in range(n):
            C[k].store(i, tmp_ptr.load[width=n_element](k * n_element).reduce_add())

    parallelize[compute_row](rows, workers)

@always_inline
fn matmul(C: UnsafePointer[element_type, MutOrigin.external], A: UnsafePointer[element_type, MutOrigin.external], B: UnsafePointer[element_type, MutOrigin.external], rows: Int, cols: Int, workers: Int) raises:
    batch_matmul[1](StaticTuple[UnsafePointer[element_type, MutOrigin.external], 1](C), A, StaticTuple[UnsafePointer[element_type, MutOrigin.external], 1](B), rows, cols, workers)

@always_inline
fn add(dest: UnsafePointer[element_type, MutOrigin.external], src: UnsafePointer[element_type, MutOrigin.external], size: Int):
    @parameter
    fn add_kernel[_nelts: Int](i: Int):
        var a = dest.offset(i).load[width=_nelts](0)
        var b = src.offset(i).load[width=_nelts](0)
        dest.store[width=_nelts](i, a + b)
    vectorize[add_kernel, n_element](size)

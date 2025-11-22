"""
Mathematical operations module

Implements essential neural network math kernels such as RMSNorm,
softmax, matrix multiplication with SIMD vectorization and
parallelization for CPU backends.
"""

import math
from sys.info import simd_width_of
from memory import UnsafePointer, stack_allocation
from utils import StaticTuple
from algorithm import vectorize, parallelize

alias element_type = Float32
alias n_element = 4 * simd_width_of[element_type]()
alias BufferPtrFloat32 = UnsafePointer[element_type]


# RMS normalization kernel
@always_inline
fn rmsnorm(
    mut o: BufferPtrFloat32,
    x: BufferPtrFloat32,
    weight: BufferPtrFloat32,
    size: Int,
):
    var tmp_ptr = stack_allocation[n_element, element_type]()
    tmp_ptr.store[width=n_element](0, SIMD[element_type, n_element](0))

    @parameter
    fn _sum2[_n_element: Int](j: Int):
        var val = x.offset(j).load[width=_n_element](0) ** 2
        var curr = tmp_ptr.load[width=_n_element](0)
        tmp_ptr.store[width=_n_element](0, curr + val)

    vectorize[_sum2, n_element](size)

    var ss: element_type = tmp_ptr.load[width=n_element](0).reduce_add()
    ss = ss / size + element_type(1e-5)
    ss = 1.0 / math.sqrt(ss)

    @parameter
    fn _norm[_n_element: Int](j: Int):
        var val = (
            weight.load[width=_n_element](j) * ss * x.load[width=_n_element](j)
        )
        o.offset(j).store[width=_n_element](0, val)

    vectorize[_norm, n_element](size)


# Softmax kernels
@always_inline
fn softmax(mut x: BufferPtrFloat32, size: Int):
    softmax(x, 0, size)


@always_inline
fn softmax(mut x: BufferPtrFloat32, start: Int, end: Int):
    var max_val: element_type = -1e9

    @parameter
    fn _max[_n_element: Int](ii: Int):
        var val = x.load[width=_n_element](start + ii).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[_max, n_element](end - start)

    var acc_ptr = stack_allocation[n_element, element_type]()
    acc_ptr.store[width=n_element](0, SIMD[element_type, n_element](0))

    @parameter
    fn _exp[_n_element: Int](ii: Int):
        var val = math.exp(x.load[width=_n_element](start + ii) - max_val)
        x.store[width=_n_element](start + ii, val)
        var curr = acc_ptr.load[width=_n_element](0)
        acc_ptr.store[width=_n_element](0, curr + val)

    vectorize[_exp, n_element](end - start)

    var ssum = acc_ptr.load[width=n_element](0).reduce_add()

    @parameter
    fn _norm[_n_element: Int](ii: Int):
        x.store[width=_n_element](
            start + ii, x.load[width=_n_element](start + ii) / ssum
        )

    vectorize[_norm, n_element](end - start)


# Batch matrix multiplication kernel
@always_inline
fn batch_matmul[
    n: Int
](
    C: StaticTuple[BufferPtrFloat32, n],
    A: BufferPtrFloat32,
    B: StaticTuple[BufferPtrFloat32, n],
    rows: Int,
    cols: Int,
    workers: Int,
):
    @parameter
    fn compute_row(i: Int):
        var tmp_ptr = stack_allocation[n * n_element, element_type]()

        @parameter
        for k in range(n):
            tmp_ptr.store[width=n_element](
                k * n_element, SIMD[element_type, n_element](0)
            )
        var row_offset = i * cols

        @parameter
        fn dot[_n_element: Int](j: Int):
            var a = A.offset(j).load[width=_n_element](0)

            @parameter
            for k in range(n):
                var val = a * B[k].offset(row_offset + j).load[
                    width=_n_element
                ](0)
                var curr = tmp_ptr.load[width=_n_element](k * n_element)
                tmp_ptr.store[width=_n_element](k * n_element, curr + val)

        vectorize[dot, n_element](cols)

        @parameter
        for k in range(n):
            C[k].store(
                i, tmp_ptr.load[width=n_element](k * n_element).reduce_add()
            )

    parallelize[compute_row](rows, workers)


# Matrix multiplication wrapper
@always_inline
fn matmul(
    C: BufferPtrFloat32,
    A: BufferPtrFloat32,
    B: BufferPtrFloat32,
    rows: Int,
    cols: Int,
    workers: Int,
) raises:
    batch_matmul[1](
        StaticTuple[BufferPtrFloat32, 1](C),
        A,
        StaticTuple[BufferPtrFloat32, 1](B),
        rows,
        cols,
        workers,
    )


# Add two buffers element-wise
@always_inline
fn add(
    dest: BufferPtrFloat32,
    src: BufferPtrFloat32,
    size: Int,
):
    @parameter
    fn add_kernel[_n_element: Int](i: Int):
        var a = dest.offset(i).load[width=_n_element](0)
        var b = src.offset(i).load[width=_n_element](0)
        dest.store[width=_n_element](i, a + b)

    vectorize[add_kernel, n_element](size)

/// Matrix module
///
/// Provides a multi-dimensional Matrix struct with dynamic shape,
/// ownership management of underlying buffer, and common indexing,
/// slicing, and utility operations optimized for high-performance
/// numerical computations in Mojo.


alias element_type = DType.float32
alias BufferPtrFloat32 = UnsafePointer[element_type, MutOrigin.external]

struct Matrix(Movable):
    var data: BufferPtrFloat32
    var allocated: Int
    var dims: List[Int]

    fn __init__(out self, *dims: Int):
        self.data = UnsafePointer[element_type, MutOrigin.external]()
        self.allocated = 0
        self.dims = List[Int]()
        for i in range(len(dims)):
            self.dims.append(dims[i])
        self.alloc()

    fn __init__(out self, ptr: BufferPtrFloat32, *dims: Int):
        self.data = ptr
        self.allocated = 0
        self.dims = List[Int]()
        for i in range(len(dims)):
            self.dims.append(dims[i])

    fn __init__(out self, ptr: BufferPtrFloat32, var dims: List[Int]):
        self.data = ptr
        self.allocated = 0
        self.dims = dims^

    @always_inline
    fn alloc(mut self, fill: Int = 0):
        self.data = alloc[element_type](self.size())
        self.allocated = 1
        if fill == 1:
            self.zero()

    @always_inline
    fn size(self) -> Int:
        var s = 1
        for i in range(len(self.dims)):
            s *= self.dims[i]
        return s

    @always_inline
    fn zero(mut self):
        memset_zero(self.data, self.size())

    @always_inline
    fn __getitem__(self, x: Int) -> element_type:
        return self.data[x]

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> element_type:
        return self.data[y * self.cols() + x]

    @always_inline
    fn __getitem__(self, z: Int, y: Int, x: Int) -> element_type:
        return self.data[z * (self.rows() * self.cols()) + y * self.cols() + x]

    @always_inline
    fn __setitem__(self, x: Int, val: element_type):
        self.data[x] = val

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: element_type):
        self.data[y * self.cols() + x] = val

    @always_inline
    fn rank(self) -> Int:
        return len(self.dims)

    @always_inline
    fn rows(self) -> Int:
        if len(self.dims) > 1:
            return self.dims[len(self.dims) - 2]
        return 1

    @always_inline
    fn cols(self) -> Int:
        if len(self.dims) > 0:
            return self.dims[len(self.dims) -1]
        return 1

    @always_inline
    fn slice(self, idx: Int) -> BufferPtrFloat32:
        if len(self.dims) > 2:
             var stride = self.rows() * self.cols()
             return self.data.offset(idx * stride)
        elif len(self.dims) > 1:
             return self.data.offset(idx * self.cols())
        else:
             return self.data.offset(idx)

    @always_inline
    fn slice(self, idx1: Int, idx2: Int) -> BufferPtrFloat32:
        var offset = idx1 * self.rows() * self.cols() + idx2 * self.cols()
        return self.data.offset(offset)

    fn _dbg_has_nan(self, n: Int) -> Bool:
        var check_count = min(n, self.size())
        for i in range(check_count):
            if not (self.data[i] == self.data[i]):
                return True
        return False

    fn dim(self, idx: Int) -> Int:
        if idx < len(self.dims):
            return self.dims[idx]
        return 0
        
    fn num_elements(self) -> Int:
        return self.size()

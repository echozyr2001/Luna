///|
/// Core NDArray structure - the foundation of Luna library
// NDArray represents an n-dimensional array with explicit shape and strides
pub struct NDArray[T] {
  data : Array[T] // Flat array storing the actual data
  shape : Array[Int] // Shape of each dimension
  strides : Array[Int] // Stride for each dimension (for memory efficiency)
  size : Int // Total number of elements
}

///|
/// Memory layout and indexing utilities
// Calculate the flat index from n-dimensional indices
fn calculate_index(indices : Array[Int], strides : Array[Int]) -> Int {
  let mut index = 0
  for i = 0; i < indices.length(); i = i + 1 {
    index = index + indices[i] * strides[i]
  }
  index
}

// Calculate strides from shape (row-major order, C-style)

///|
fn calculate_strides(shape : Array[Int]) -> Array[Int] {
  let ndim = shape.length()
  let strides = Array::make(ndim, 1)
  for i = ndim - 2; i >= 0; i = i - 1 {
    strides[i] = strides[i + 1] * shape[i + 1]
  }
  strides
}

// Calculate total size from shape

///|
fn calculate_size(shape : Array[Int]) -> Int {
  let mut size = 1
  for i = 0; i < shape.length(); i = i + 1 {
    size = size * shape[i]
  }
  size
}

///|
/// Core array creation functions
// Create array filled with zeros
pub fn zeros(shape : Array[Int]) -> NDArray[Double] {
  let size = calculate_size(shape)
  let strides = calculate_strides(shape)
  let data = Array::make(size, 0.0)
  { data, shape, strides, size }
}

// Create array filled with ones

///|
pub fn ones(shape : Array[Int]) -> NDArray[Double] {
  let size = calculate_size(shape)
  let strides = calculate_strides(shape)
  let data = Array::make(size, 1.0)
  { data, shape, strides, size }
}

// Create array filled with a specific value

///|
pub fn[T] full(shape : Array[Int], value : T) -> NDArray[T] {
  let size = calculate_size(shape)
  let strides = calculate_strides(shape)
  let data = Array::make(size, value)
  { data, shape, strides, size }
}

// Create array from flat data with specified shape

///|
pub fn[T] from_array(data : Array[T], shape : Array[Int]) -> NDArray[T] {
  let size = calculate_size(shape)
  if data.length() != size {
    abort("Data size does not match shape")
  }
  let strides = calculate_strides(shape)
  { data, shape, strides, size }
}

///|
/// Element access and modification
// Get element at n-dimensional indices
pub fn[T] get(self : NDArray[T], indices : Array[Int]) -> T {
  if indices.length() != self.shape.length() {
    abort("Number of indices must match array dimensions")
  }
  let flat_index = calculate_index(indices, self.strides)
  self.data[flat_index]
}

// Set element at n-dimensional indices

///|
pub fn[T] set(self : NDArray[T], indices : Array[Int], value : T) -> Unit {
  if indices.length() != self.shape.length() {
    abort("Number of indices must match array dimensions")
  }
  let flat_index = calculate_index(indices, self.strides)
  self.data[flat_index] = value
}

///|
/// Basic properties
pub fn[T] shape(self : NDArray[T]) -> Array[Int] {
  self.shape
}

///|
pub fn[T] size(self : NDArray[T]) -> Int {
  self.size
}

///|
pub fn[T] ndim(self : NDArray[T]) -> Int {
  self.shape.length()
}

///|
/// Additional array creation functions
// Create 1D array (similar to numpy.arange)
pub fn arange(start : Int, stop : Int, step : Int) -> NDArray[Int] {
  let data = Array::new()
  let mut i = start
  while i < stop {
    data.push(i)
    i = i + step
  }
  let shape = [data.length()]
  from_array(data, shape)
}

// Create identity matrix

///|
pub fn eye(n : Int) -> NDArray[Double] {
  let arr = zeros([n, n])
  for i = 0; i < n; i = i + 1 {
    arr.set([i, i], 1.0)
  }
  arr
}

// Create array with random values (simple linear congruential generator)

///|
pub fn random(shape : Array[Int], seed : Int) -> NDArray[Double] {
  let size = calculate_size(shape)
  let strides = calculate_strides(shape)
  let data = Array::make(size, 0.0)
  let mut rng_state = seed
  for i = 0; i < size; i = i + 1 {
    // Simple LCG: next = (a * current + c) % m
    rng_state = (1664525 * rng_state + 1013904223) % 2147483647
    data[i] = (rng_state % 10000).to_double() / 10000.0 // Scale to [0, 1)
  }
  { data, shape, strides, size }
}

///|
/// Element-wise operations (the key differentiator - operations are element-wise by default)
// Element-wise addition
pub fn add(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  // For now, require same shape (broadcasting will be added later)
  if a.shape.length() != b.shape.length() {
    abort(
      "Arrays must have same number of dimensions for element-wise operations",
    )
  }
  for i = 0; i < a.shape.length(); i = i + 1 {
    if a.shape[i] != b.shape[i] {
      abort("Arrays must have same shape for element-wise operations")
    }
  }
  let result_data = Array::make(a.size, a.data[0] + b.data[0])
  for i = 0; i < a.size; i = i + 1 {
    result_data[i] = a.data[i] + b.data[i]
  }
  { data: result_data, shape: a.shape, strides: a.strides, size: a.size }
}

// Element-wise subtraction

///|
pub fn sub(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  if a.shape.length() != b.shape.length() {
    abort(
      "Arrays must have same number of dimensions for element-wise operations",
    )
  }
  for i = 0; i < a.shape.length(); i = i + 1 {
    if a.shape[i] != b.shape[i] {
      abort("Arrays must have same shape for element-wise operations")
    }
  }
  let result_data = Array::make(a.size, a.data[0] - b.data[0])
  for i = 0; i < a.size; i = i + 1 {
    result_data[i] = a.data[i] - b.data[i]
  }
  { data: result_data, shape: a.shape, strides: a.strides, size: a.size }
}

// Element-wise multiplication (NOT matrix multiplication)

///|
pub fn mul(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  if a.shape.length() != b.shape.length() {
    abort(
      "Arrays must have same number of dimensions for element-wise operations",
    )
  }
  for i = 0; i < a.shape.length(); i = i + 1 {
    if a.shape[i] != b.shape[i] {
      abort("Arrays must have same shape for element-wise operations")
    }
  }
  let result_data = Array::make(a.size, a.data[0] * b.data[0])
  for i = 0; i < a.size; i = i + 1 {
    result_data[i] = a.data[i] * b.data[i]
  }
  { data: result_data, shape: a.shape, strides: a.strides, size: a.size }
}

// Scalar operations

///|
pub fn add_scalar(a : NDArray[Double], scalar : Double) -> NDArray[Double] {
  let result_data = Array::make(a.size, a.data[0] + scalar)
  for i = 0; i < a.size; i = i + 1 {
    result_data[i] = a.data[i] + scalar
  }
  { data: result_data, shape: a.shape, strides: a.strides, size: a.size }
}

///|
pub fn mul_scalar(a : NDArray[Double], scalar : Double) -> NDArray[Double] {
  let result_data = Array::make(a.size, a.data[0] * scalar)
  for i = 0; i < a.size; i = i + 1 {
    result_data[i] = a.data[i] * scalar
  }
  { data: result_data, shape: a.shape, strides: a.strides, size: a.size }
}

///|
/// Matrix operations (explicit, not element-wise)
// Matrix multiplication (dot product)
pub fn dot(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  // Support 1D and 2D arrays
  match (a.shape.length(), b.shape.length()) {
    (1, 1) => {
      // Vector dot product
      if a.shape[0] != b.shape[0] {
        abort("Vector dimensions must match for dot product")
      }
      let mut result = 0.0
      for i = 0; i < a.shape[0]; i = i + 1 {
        result = result + a.data[i] * b.data[i]
      }
      from_array([result], [1])
    }
    (2, 2) => {
      // Matrix multiplication
      let (m, k) = (a.shape[0], a.shape[1])
      let (k2, n) = (b.shape[0], b.shape[1])
      if k != k2 {
        abort("Matrix inner dimensions must match for multiplication")
      }
      let result = zeros([m, n])
      for i = 0; i < m; i = i + 1 {
        for j = 0; j < n; j = j + 1 {
          let mut sum = 0.0
          for l = 0; l < k; l = l + 1 {
            let a_val = a.get([i, l])
            let b_val = b.get([l, j])
            sum = sum + a_val * b_val
          }
          result.set([i, j], sum)
        }
      }
      result
    }
    (2, 1) => {
      // Matrix-vector multiplication
      let (m, k) = (a.shape[0], a.shape[1])
      if k != b.shape[0] {
        abort("Matrix columns must match vector length")
      }
      let result = zeros([m])
      for i = 0; i < m; i = i + 1 {
        let mut sum = 0.0
        for j = 0; j < k; j = j + 1 {
          let a_val = a.get([i, j])
          let b_val = b.get([j])
          sum = sum + a_val * b_val
        }
        result.set([i], sum)
      }
      result
    }
    _ => abort("Unsupported array dimensions for dot product")
  }
}

// Transpose (2D arrays only for now)

///|
pub fn transpose(a : NDArray[Double]) -> NDArray[Double] {
  if a.shape.length() != 2 {
    abort("Transpose only supported for 2D arrays")
  }
  let (m, n) = (a.shape[0], a.shape[1])
  let result = zeros([n, m])
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let val = a.get([i, j])
      result.set([j, i], val)
    }
  }
  result
}

///|
/// Aggregation functions
// Sum all elements
pub fn sum(a : NDArray[Double]) -> Double {
  let mut result = 0.0
  for i = 0; i < a.size; i = i + 1 {
    result = result + a.data[i]
  }
  result
}

// Mean of all elements

///|
pub fn mean(a : NDArray[Double]) -> Double {
  sum(a) / a.size.to_double()
}

// Maximum element

///|
pub fn max(a : NDArray[Double]) -> Double {
  let mut result = a.data[0]
  for i = 1; i < a.size; i = i + 1 {
    if a.data[i] > result {
      result = a.data[i]
    }
  }
  result
}

// Minimum element

///|
pub fn min(a : NDArray[Double]) -> Double {
  let mut result = a.data[0]
  for i = 1; i < a.size; i = i + 1 {
    if a.data[i] < result {
      result = a.data[i]
    }
  }
  result
}

///|
/// Utility functions
// Reshape array (must preserve total size)
pub fn[T] reshape(a : NDArray[T], new_shape : Array[Int]) -> NDArray[T] {
  let new_size = calculate_size(new_shape)
  if new_size != a.size {
    abort("Cannot reshape array: size mismatch")
  }
  let new_strides = calculate_strides(new_shape)
  { data: a.data, shape: new_shape, strides: new_strides, size: a.size }
}

// Flatten to 1D array

///|
pub fn[T] flatten(a : NDArray[T]) -> NDArray[T] {
  reshape(a, [a.size])
}

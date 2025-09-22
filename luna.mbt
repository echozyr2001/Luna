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
// Normalize negative indices to positive ones
fn normalize_index(index : Int, dim_size : Int) -> Int {
  if index < 0 {
    if index + dim_size < 0 {
      abort("Index out of bounds")
    }
    index + dim_size
  } else {
    if index >= dim_size {
      abort("Index out of bounds")
    }
    index
  }
}

// Get element at n-dimensional indices (supports negative indices)
pub fn[T] get(self : NDArray[T], indices : Array[Int]) -> T {
  if indices.length() != self.shape.length() {
    abort("Number of indices must match array dimensions")
  }
  let normalized_indices = Array::make(indices.length(), 0)
  for i = 0; i < indices.length(); i = i + 1 {
    normalized_indices[i] = normalize_index(indices[i], self.shape[i])
  }
  let flat_index = calculate_index(normalized_indices, self.strides)
  self.data[flat_index]
}

// Set element at n-dimensional indices

///|
pub fn[T] set(self : NDArray[T], indices : Array[Int], value : T) -> Unit {
  if indices.length() != self.shape.length() {
    abort("Number of indices must match array dimensions")
  }
  let normalized_indices = Array::make(indices.length(), 0)
  for i = 0; i < indices.length(); i = i + 1 {
    normalized_indices[i] = normalize_index(indices[i], self.shape[i])
  }
  let flat_index = calculate_index(normalized_indices, self.strides)
  self.data[flat_index] = value
}

///|
/// Advanced indexing and slicing
// Slice represents a slice with start:stop:step
pub struct Slice {
  start : Int?
  stop : Int?
  step : Int
}

// Create a basic slice (start:stop)
pub fn slice(start : Int?, stop : Int?) -> Slice {
  { start, stop, step: 1 }
}

// Create a slice with step (start:stop:step)
pub fn slice_step(start : Int?, stop : Int?, step : Int) -> Slice {
  { start, stop, step }
}

// Resolve slice bounds for a given dimension size
fn resolve_slice(slice : Slice, dim_size : Int) -> (Int, Int, Int) {
  let step = slice.step
  if step == 0 {
    abort("Slice step cannot be zero")
  }

  let start = match slice.start {
    Some(s) => normalize_index(s, dim_size)
    None => if step > 0 { 0 } else { dim_size - 1 }
  }

  let stop = match slice.stop {
    Some(s) => {
      // Handle negative stop indices properly
      if s < 0 {
        let normalized = s + dim_size
        if normalized < 0 { 0 } else { normalized }
      } else {
        if s > dim_size { dim_size } else { s }
      }
    }
    None => if step > 0 { dim_size } else { -1 }
  }

  (start, stop, step)
}

// Get subarray using slices for each dimension (simplified implementation)
pub fn[T] slice_array(self : NDArray[T], slices : Array[Slice]) -> NDArray[T] {
  if slices.length() != self.shape.length() {
    abort("Number of slices must match array dimensions")
  }

  // Calculate new shape
  let new_shape = Array::new()

  for i = 0; i < slices.length(); i = i + 1 {
    let (start, stop, step) = resolve_slice(slices[i], self.shape[i])
    let dim_size = if step > 0 {
      if stop <= start { 0 } else { (stop - start + step - 1) / step }
    } else {
      if stop >= start { 0 } else { (start - stop - step - 1) / (-step) }
    }
    new_shape.push(dim_size)
  }

  let new_size = calculate_size(new_shape)

  // For empty result, return immediately
  if new_size == 0 {
    let new_strides = calculate_strides(new_shape)
    let empty_data = Array::new()
    return { data: empty_data, shape: new_shape, strides: new_strides, size: 0 }
  }

  let new_data = Array::make(new_size, self.data[0])

  // Simple approach: iterate through all result positions
  for flat_idx = 0; flat_idx < new_size; flat_idx = flat_idx + 1 {
    // Convert flat index to multi-dimensional indices
    let result_indices = Array::make(new_shape.length(), 0)
    let mut temp_idx = flat_idx

    for dim = new_shape.length() - 1; dim >= 0; dim = dim - 1 {
      result_indices[dim] = temp_idx % new_shape[dim]
      temp_idx = temp_idx / new_shape[dim]
    }

    // Convert result indices to source indices
    let src_indices = Array::make(self.shape.length(), 0)
    for i = 0; i < slices.length(); i = i + 1 {
      let (start, _, step) = resolve_slice(slices[i], self.shape[i])
      src_indices[i] = start + result_indices[i] * step
    }

    new_data[flat_idx] = self.get(src_indices)
  }

  let new_strides = calculate_strides(new_shape)
  { data: new_data, shape: new_shape, strides: new_strides, size: new_size }
}

// Fancy indexing - select elements using arrays of indices
pub fn[T] fancy_index(self : NDArray[T], indices : Array[Array[Int]]) -> NDArray[T] {
  if indices.length() != self.shape.length() {
    abort("Number of index arrays must match array dimensions")
  }

  // All index arrays must have the same length
  let result_size = indices[0].length()

  for i = 1; i < indices.length(); i = i + 1 {
    if indices[i].length() != result_size {
      abort("All index arrays must have the same length")
    }
  }

  let result_data = Array::make(result_size, self.data[0])

  for i = 0; i < result_size; i = i + 1 {
    let coords = Array::make(self.shape.length(), 0)
    for j = 0; j < indices.length(); j = j + 1 {
      coords[j] = normalize_index(indices[j][i], self.shape[j])
    }
    let flat_index = calculate_index(coords, self.strides)
    result_data[i] = self.data[flat_index]
  }

  from_array(result_data, [result_size])
}

// Boolean indexing - select elements where condition is true
pub fn[T] boolean_index(self : NDArray[T], mask : NDArray[Bool]) -> NDArray[T] {
  if self.size != mask.size {
    abort("Mask array must have the same size as the source array")
  }

  // Count true values to determine result size
  let mut count = 0
  for i = 0; i < mask.size; i = i + 1 {
    if mask.data[i] {
      count = count + 1
    }
  }

  let result_data = Array::make(count, self.data[0])
  let mut result_idx = 0

  for i = 0; i < self.size; i = i + 1 {
    if mask.data[i] {
      result_data[result_idx] = self.data[i]
      result_idx = result_idx + 1
    }
  }

  from_array(result_data, [count])
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
/// Broadcasting mechanism for NumPy-compatible operations
// Check if two shapes can be broadcast together
fn can_broadcast(shape1 : Array[Int], shape2 : Array[Int]) -> Bool {
  let ndim1 = shape1.length()
  let ndim2 = shape2.length()
  let max_ndim = if ndim1 > ndim2 { ndim1 } else { ndim2 }

  for i = 0; i < max_ndim; i = i + 1 {
    let dim1 = if i < ndim1 { shape1[ndim1 - 1 - i] } else { 1 }
    let dim2 = if i < ndim2 { shape2[ndim2 - 1 - i] } else { 1 }

    if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
      return false
    }
  }
  true
}

// Calculate the broadcast shape for two arrays
fn broadcast_shape(shape1 : Array[Int], shape2 : Array[Int]) -> Array[Int] {
  if not(can_broadcast(shape1, shape2)) {
    abort("Shapes cannot be broadcast together")
  }

  let ndim1 = shape1.length()
  let ndim2 = shape2.length()
  let max_ndim = if ndim1 > ndim2 { ndim1 } else { ndim2 }
  let result = Array::make(max_ndim, 1)

  for i = 0; i < max_ndim; i = i + 1 {
    let dim1 = if i < ndim1 { shape1[ndim1 - 1 - i] } else { 1 }
    let dim2 = if i < ndim2 { shape2[ndim2 - 1 - i] } else { 1 }
    result[max_ndim - 1 - i] = if dim1 > dim2 { dim1 } else { dim2 }
  }
  result
}

// Get broadcasted index for an array given the broadcast shape
fn get_broadcast_index(
  original_shape : Array[Int],
  broadcast_shape : Array[Int],
  broadcast_indices : Array[Int]
) -> Array[Int] {
  let ndim_orig = original_shape.length()
  let ndim_broadcast = broadcast_shape.length()
  let result = Array::make(ndim_orig, 0)

  for i = 0; i < ndim_orig; i = i + 1 {
    let broadcast_idx = ndim_broadcast - ndim_orig + i
    let dim_size = original_shape[i]

    if dim_size == 1 {
      result[i] = 0  // Broadcasting: use index 0 for size-1 dimensions
    } else {
      result[i] = broadcast_indices[broadcast_idx]
    }
  }
  result
}

///|
/// Element-wise operations with broadcasting support
// Element-wise addition with broadcasting support
pub fn add(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  if not(can_broadcast(a.shape, b.shape)) {
    abort("Arrays cannot be broadcast together for addition")
  }

  let result_shape = broadcast_shape(a.shape, b.shape)
  let result_size = calculate_size(result_shape)
  let result_strides = calculate_strides(result_shape)
  let result_data = Array::make(result_size, 0.0)

  // Generate all possible indices for the result array
  let indices = Array::make(result_shape.length(), 0)

  fn iterate_indices(
    indices : Array[Int], shape : Array[Int], dim : Int,
    a_arr : NDArray[Double], b_arr : NDArray[Double],
    result : Array[Double], result_strides : Array[Int]
  ) -> Unit {
    if dim == shape.length() {
      // Calculate indices for both input arrays
      let a_indices = get_broadcast_index(a_arr.shape, shape, indices)
      let b_indices = get_broadcast_index(b_arr.shape, shape, indices)

      let a_val = a_arr.get(a_indices)
      let b_val = b_arr.get(b_indices)
      let result_idx = calculate_index(indices, result_strides)
      result[result_idx] = a_val + b_val
      return
    }

    for i = 0; i < shape[dim]; i = i + 1 {
      indices[dim] = i
      iterate_indices(indices, shape, dim + 1, a_arr, b_arr, result, result_strides)
    }
  }

  iterate_indices(indices, result_shape, 0, a, b, result_data, result_strides)
  { data: result_data, shape: result_shape, strides: result_strides, size: result_size }
}

// Element-wise subtraction

///|
// Element-wise subtraction with broadcasting support
pub fn sub(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  if not(can_broadcast(a.shape, b.shape)) {
    abort("Arrays cannot be broadcast together for subtraction")
  }

  let result_shape = broadcast_shape(a.shape, b.shape)
  let result_size = calculate_size(result_shape)
  let result_strides = calculate_strides(result_shape)
  let result_data = Array::make(result_size, 0.0)

  let indices = Array::make(result_shape.length(), 0)

  fn iterate_indices(
    indices : Array[Int], shape : Array[Int], dim : Int,
    a_arr : NDArray[Double], b_arr : NDArray[Double],
    result : Array[Double], result_strides : Array[Int]
  ) -> Unit {
    if dim == shape.length() {
      let a_indices = get_broadcast_index(a_arr.shape, shape, indices)
      let b_indices = get_broadcast_index(b_arr.shape, shape, indices)

      let a_val = a_arr.get(a_indices)
      let b_val = b_arr.get(b_indices)
      let result_idx = calculate_index(indices, result_strides)
      result[result_idx] = a_val - b_val
      return
    }

    for i = 0; i < shape[dim]; i = i + 1 {
      indices[dim] = i
      iterate_indices(indices, shape, dim + 1, a_arr, b_arr, result, result_strides)
    }
  }

  iterate_indices(indices, result_shape, 0, a, b, result_data, result_strides)
  { data: result_data, shape: result_shape, strides: result_strides, size: result_size }
}

// Element-wise multiplication with broadcasting support
pub fn mul(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  if not(can_broadcast(a.shape, b.shape)) {
    abort("Arrays cannot be broadcast together for multiplication")
  }

  let result_shape = broadcast_shape(a.shape, b.shape)
  let result_size = calculate_size(result_shape)
  let result_strides = calculate_strides(result_shape)
  let result_data = Array::make(result_size, 0.0)

  let indices = Array::make(result_shape.length(), 0)

  fn iterate_indices(
    indices : Array[Int], shape : Array[Int], dim : Int,
    a_arr : NDArray[Double], b_arr : NDArray[Double],
    result : Array[Double], result_strides : Array[Int]
  ) -> Unit {
    if dim == shape.length() {
      let a_indices = get_broadcast_index(a_arr.shape, shape, indices)
      let b_indices = get_broadcast_index(b_arr.shape, shape, indices)

      let a_val = a_arr.get(a_indices)
      let b_val = b_arr.get(b_indices)
      let result_idx = calculate_index(indices, result_strides)
      result[result_idx] = a_val * b_val
      return
    }

    for i = 0; i < shape[dim]; i = i + 1 {
      indices[dim] = i
      iterate_indices(indices, shape, dim + 1, a_arr, b_arr, result, result_strides)
    }
  }

  iterate_indices(indices, result_shape, 0, a, b, result_data, result_strides)
  { data: result_data, shape: result_shape, strides: result_strides, size: result_size }
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
/// Apache Arrow integration foundation
// Arrow-compatible columnar memory layout
pub struct ArrowBuffer[T] {
  data : Array[T]        // Raw data buffer
  null_bitmap : Array[Bool]?  // Optional null bitmap for nullable data
  length : Int           // Number of elements
  offset : Int           // Starting offset in the buffer
}

// Create Arrow buffer from array data
pub fn[T] arrow_buffer(data : Array[T]) -> ArrowBuffer[T] {
  { data, null_bitmap: None, length: data.length(), offset: 0 }
}

// Create Arrow buffer with null bitmap
pub fn[T] arrow_buffer_nullable(data : Array[T], null_bitmap : Array[Bool]) -> ArrowBuffer[T] {
  if data.length() != null_bitmap.length() {
    abort("Data and null bitmap must have the same length")
  }
  { data, null_bitmap: Some(null_bitmap), length: data.length(), offset: 0 }
}

// Arrow-compatible array structure (columnar format)
pub struct ArrowArray[T] {
  buffer : ArrowBuffer[T]
  name : String          // Column name
}

// Create Arrow array from NDArray (converts to columnar format)
pub fn[T] to_arrow_array(arr : NDArray[T], name : String) -> ArrowArray[T] {
  // Flatten the NDArray to 1D for columnar storage
  let flattened = flatten(arr)
  let buffer = arrow_buffer(flattened.data)
  { buffer, name }
}

// Convert Arrow array back to NDArray with specified shape
pub fn[T] from_arrow_array(arrow_arr : ArrowArray[T], shape : Array[Int]) -> NDArray[T] {
  let expected_size = calculate_size(shape)
  if arrow_arr.buffer.length != expected_size {
    abort("Arrow buffer size does not match expected shape")
  }
  from_array(arrow_arr.buffer.data, shape)
}

// Table structure for columnar data (like Arrow Table)
pub struct ArrowTable {
  columns : Array[ArrowArray[Double]]  // For now, focus on Double columns
  num_rows : Int
}

// Create Arrow table from multiple NDArrays
pub fn create_arrow_table(arrays : Array[NDArray[Double]], names : Array[String]) -> ArrowTable {
  if arrays.length() != names.length() {
    abort("Number of arrays must match number of column names")
  }

  let columns = Array::new()
  let mut total_elements = 0

  for i = 0; i < arrays.length(); i = i + 1 {
    let arrow_col = to_arrow_array(arrays[i], names[i])
    columns.push(arrow_col)
    if i == 0 {
      total_elements = arrays[i].size
    } else if arrays[i].size != total_elements {
      abort("All arrays must have the same number of elements for table creation")
    }
  }

  { columns, num_rows: total_elements }
}

// Extract column from Arrow table
pub fn get_column(table : ArrowTable, column_name : String) -> ArrowArray[Double]? {
  for i = 0; i < table.columns.length(); i = i + 1 {
    if table.columns[i].name == column_name {
      return Some(table.columns[i])
    }
  }
  None
}

// Zero-copy column selection (returns view of existing data)
pub fn select_columns(table : ArrowTable, column_names : Array[String]) -> ArrowTable {
  let selected_columns = Array::new()

  for name in column_names {
    match get_column(table, name) {
      Some(col) => selected_columns.push(col)
      None => abort("Column '" + name + "' not found in table")
    }
  }

  { columns: selected_columns, num_rows: table.num_rows }
}

// Memory-efficient row iteration (columnar access pattern)
pub fn[T] iterate_rows(table : ArrowTable, f : (Array[Double]) -> T) -> Array[T] {
  if table.columns.length() == 0 {
    return Array::new()
  }

  let results = Array::new()
  for row_idx = 0; row_idx < table.num_rows; row_idx = row_idx + 1 {
    let row_values = Array::new()
    for col in table.columns {
      row_values.push(col.buffer.data[row_idx])
    }
    results.push(f(row_values))
  }
  results
}

///|
/// SIMD-optimized operations foundation
// SIMD-friendly block size for vectorized operations
let simd_block_size = 8

// Vectorized addition (SIMD-optimized when available)
pub fn add_vectorized(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  if not(can_broadcast(a.shape, b.shape)) {
    abort("Arrays cannot be broadcast together for vectorized addition")
  }

  // For same-shape arrays, use optimized path
  if a.shape.length() == b.shape.length() {
    let mut same_shape = true
    for i = 0; i < a.shape.length(); i = i + 1 {
      if a.shape[i] != b.shape[i] {
        same_shape = false
        break
      }
    }

    if same_shape {
      return add_same_shape_vectorized(a, b)
    }
  }

  // Fall back to broadcasting version
  add(a, b)
}

// Optimized addition for same-shape arrays
fn add_same_shape_vectorized(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  let result_data = Array::make(a.size, 0.0)

  // Process in SIMD-friendly blocks
  let full_blocks = a.size / simd_block_size
  let remainder = a.size % simd_block_size

  // Process full blocks (SIMD-optimizable)
  for block = 0; block < full_blocks; block = block + 1 {
    let start_idx = block * simd_block_size

    // Unrolled loop for better vectorization
    result_data[start_idx] = a.data[start_idx] + b.data[start_idx]
    result_data[start_idx + 1] = a.data[start_idx + 1] + b.data[start_idx + 1]
    result_data[start_idx + 2] = a.data[start_idx + 2] + b.data[start_idx + 2]
    result_data[start_idx + 3] = a.data[start_idx + 3] + b.data[start_idx + 3]
    result_data[start_idx + 4] = a.data[start_idx + 4] + b.data[start_idx + 4]
    result_data[start_idx + 5] = a.data[start_idx + 5] + b.data[start_idx + 5]
    result_data[start_idx + 6] = a.data[start_idx + 6] + b.data[start_idx + 6]
    result_data[start_idx + 7] = a.data[start_idx + 7] + b.data[start_idx + 7]
  }

  // Handle remainder
  let remainder_start = full_blocks * simd_block_size
  for i = 0; i < remainder; i = i + 1 {
    let idx = remainder_start + i
    result_data[idx] = a.data[idx] + b.data[idx]
  }

  { data: result_data, shape: a.shape, strides: a.strides, size: a.size }
}

// Vectorized multiplication
pub fn mul_vectorized(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  if not(can_broadcast(a.shape, b.shape)) {
    abort("Arrays cannot be broadcast together for vectorized multiplication")
  }

  // For same-shape arrays, use optimized path
  if a.shape.length() == b.shape.length() {
    let mut same_shape = true
    for i = 0; i < a.shape.length(); i = i + 1 {
      if a.shape[i] != b.shape[i] {
        same_shape = false
        break
      }
    }

    if same_shape {
      return mul_same_shape_vectorized(a, b)
    }
  }

  // Fall back to broadcasting version
  mul(a, b)
}

// Optimized multiplication for same-shape arrays
fn mul_same_shape_vectorized(a : NDArray[Double], b : NDArray[Double]) -> NDArray[Double] {
  let result_data = Array::make(a.size, 0.0)

  let full_blocks = a.size / simd_block_size
  let remainder = a.size % simd_block_size

  // Process full blocks with unrolled loop
  for block = 0; block < full_blocks; block = block + 1 {
    let start_idx = block * simd_block_size

    result_data[start_idx] = a.data[start_idx] * b.data[start_idx]
    result_data[start_idx + 1] = a.data[start_idx + 1] * b.data[start_idx + 1]
    result_data[start_idx + 2] = a.data[start_idx + 2] * b.data[start_idx + 2]
    result_data[start_idx + 3] = a.data[start_idx + 3] * b.data[start_idx + 3]
    result_data[start_idx + 4] = a.data[start_idx + 4] * b.data[start_idx + 4]
    result_data[start_idx + 5] = a.data[start_idx + 5] * b.data[start_idx + 5]
    result_data[start_idx + 6] = a.data[start_idx + 6] * b.data[start_idx + 6]
    result_data[start_idx + 7] = a.data[start_idx + 7] * b.data[start_idx + 7]
  }

  // Handle remainder
  let remainder_start = full_blocks * simd_block_size
  for i = 0; i < remainder; i = i + 1 {
    let idx = remainder_start + i
    result_data[idx] = a.data[idx] * b.data[idx]
  }

  { data: result_data, shape: a.shape, strides: a.strides, size: a.size }
}

// Vectorized sum reduction
pub fn sum_vectorized(a : NDArray[Double]) -> Double {
  let mut result = 0.0
  let full_blocks = a.size / simd_block_size
  let remainder = a.size % simd_block_size

  // Process full blocks with loop unrolling
  for block = 0; block < full_blocks; block = block + 1 {
    let start_idx = block * simd_block_size

    // Unrolled accumulation for better performance
    result = result + a.data[start_idx]
    result = result + a.data[start_idx + 1]
    result = result + a.data[start_idx + 2]
    result = result + a.data[start_idx + 3]
    result = result + a.data[start_idx + 4]
    result = result + a.data[start_idx + 5]
    result = result + a.data[start_idx + 6]
    result = result + a.data[start_idx + 7]
  }

  // Handle remainder
  let remainder_start = full_blocks * simd_block_size
  for i = 0; i < remainder; i = i + 1 {
    result = result + a.data[remainder_start + i]
  }

  result
}

// Fused multiply-add operation (common in ML)
pub fn fma_vectorized(a : NDArray[Double], b : NDArray[Double], c : NDArray[Double]) -> NDArray[Double] {
  // Check shapes are compatible
  if not(can_broadcast(a.shape, b.shape)) || not(can_broadcast(a.shape, c.shape)) {
    abort("Arrays cannot be broadcast together for FMA operation")
  }

  // For now, simple implementation - could be optimized with actual FMA instructions
  let temp = mul_vectorized(a, b)
  add_vectorized(temp, c)
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

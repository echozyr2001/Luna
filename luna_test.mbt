///|
/// Luna NDArray tests
test "array_creation" {
  // Test zeros
  let arr = zeros([2, 3])
  inspect(arr.shape(), content="[2, 3]")
  inspect(arr.size(), content="6")
  inspect(arr.ndim(), content="2")
}

///|
test "array_from_data" {
  let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let arr = from_array(data, [2, 3])
  inspect(arr.shape(), content="[2, 3]")
  inspect(arr.get([0, 0]), content="1")
  inspect(arr.get([0, 1]), content="2")
  inspect(arr.get([1, 2]), content="6")
}

///|
test "ones_and_full" {
  let ones_arr = ones([2, 2])
  inspect(ones_arr.get([0, 0]), content="1")
  inspect(ones_arr.get([1, 1]), content="1")
  let full_arr = full([2, 2], 5.0)
  inspect(full_arr.get([0, 0]), content="5")
  inspect(full_arr.get([1, 1]), content="5")
}

///|
test "arange_function" {
  let arr = arange(0, 10, 2)
  inspect(arr.shape(), content="[5]")
  inspect(arr.get([0]), content="0")
  inspect(arr.get([1]), content="2")
  inspect(arr.get([4]), content="8")
}

///|
test "identity_matrix" {
  let eye_arr = eye(3)
  inspect(eye_arr.shape(), content="[3, 3]")
  inspect(eye_arr.get([0, 0]), content="1")
  inspect(eye_arr.get([1, 1]), content="1")
  inspect(eye_arr.get([2, 2]), content="1")
  inspect(eye_arr.get([0, 1]), content="0")
  inspect(eye_arr.get([1, 0]), content="0")
}

///|
test "element_wise_operations" {
  let a = from_array([1.0, 2.0, 3.0, 4.0], [2, 2])
  let b = from_array([2.0, 2.0, 2.0, 2.0], [2, 2])
  let sum_result = add(a, b)
  inspect(sum_result.get([0, 0]), content="3")
  inspect(sum_result.get([1, 1]), content="6")
  let mul_result = mul(a, b)
  inspect(mul_result.get([0, 0]), content="2")
  inspect(mul_result.get([1, 1]), content="8")
}

///|
test "scalar_operations" {
  let a = from_array([1.0, 2.0, 3.0, 4.0], [2, 2])
  let add_scalar_result = add_scalar(a, 10.0)
  inspect(add_scalar_result.get([0, 0]), content="11")
  inspect(add_scalar_result.get([1, 1]), content="14")
  let mul_scalar_result = mul_scalar(a, 3.0)
  inspect(mul_scalar_result.get([0, 0]), content="3")
  inspect(mul_scalar_result.get([1, 1]), content="12")
}

///|
test "matrix_multiplication" {
  // Test 2x2 matrix multiplication
  let a = from_array([1.0, 2.0, 3.0, 4.0], [2, 2]) // [[1, 2], [3, 4]]
  let b = from_array([5.0, 6.0, 7.0, 8.0], [2, 2]) // [[5, 6], [7, 8]]
  let result = dot(a, b) // Should be [[19, 22], [43, 50]]
  inspect(result.get([0, 0]), content="19")
  inspect(result.get([0, 1]), content="22")
  inspect(result.get([1, 0]), content="43")
  inspect(result.get([1, 1]), content="50")
}

///|
test "vector_dot_product" {
  let a = from_array([1.0, 2.0, 3.0], [3])
  let b = from_array([4.0, 5.0, 6.0], [3])
  let result = dot(a, b) // Should be 1*4 + 2*5 + 3*6 = 32
  inspect(result.get([0]), content="32")
}

///|
test "matrix_vector_multiplication" {
  let a = from_array([1.0, 2.0, 3.0, 4.0], [2, 2]) // [[1, 2], [3, 4]]
  let b = from_array([5.0, 6.0], [2]) // [5, 6]
  let result = dot(a, b) // Should be [17, 39]
  inspect(result.get([0]), content="17")
  inspect(result.get([1]), content="39")
}

///|
test "transpose_operation" {
  let a = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]) // [[1, 2, 3], [4, 5, 6]]
  let transposed = transpose(a)
  inspect(transposed.shape(), content="[3, 2]")
  inspect(transposed.get([0, 0]), content="1")
  inspect(transposed.get([0, 1]), content="4")
  inspect(transposed.get([1, 0]), content="2")
  inspect(transposed.get([2, 1]), content="6")
}

///|
test "aggregation_functions" {
  let a = from_array([1.0, 2.0, 3.0, 4.0], [2, 2])
  inspect(mean(a), content="2.5")
  inspect(max(a), content="4")
  inspect(min(a), content="1")
}

///|
test "reshape_and_flatten" {
  let a = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let reshaped = reshape(a, [3, 2])
  inspect(reshaped.shape(), content="[3, 2]")
  inspect(reshaped.get([0, 0]), content="1")
  inspect(reshaped.get([2, 1]), content="6")
  let flattened = flatten(a)
  inspect(flattened.shape(), content="[6]")
  inspect(flattened.get([5]), content="6")
}

///|
test "random_generation" {
  let arr = random([2, 2], 42)
  inspect(arr.shape(), content="[2, 2]")
  // Random values should be between 0 and 1
  let val = arr.get([0, 0])
  inspect(val >= 0.0 && val < 1.0, content="true")
}

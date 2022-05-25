// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

func.func @reshaper0(%arg0: tensor<1x1x60xf32>) -> tensor<60xf32> {
  %0 = tile.reshape %arg0 : (tensor<1x1x60xf32>) -> tensor<60xf32>
  return %0 : tensor<60xf32>
}

// CHECK-LABEL: func.func @reshaper0
// CHECK: tensor.collapse_shape
// CHECK: tensor<60xf32>

func.func @reshaper1(%arg0: tensor<2x4x5xf32>) -> tensor<2x20xf32> {
  %0 = tile.reshape %arg0 : (tensor<2x4x5xf32>) -> tensor<2x20xf32>
  return %0 : tensor<2x20xf32>
}

// CHECK-LABEL: func.func @reshaper1
// CHECK: tensor.collapse_shape
// CHECK: tensor<2x20xf32>

func.func @reshaper2(%arg1: tensor<5x2x3xf32>) -> tensor<5x6xf32> {
  %0 = tile.reshape %arg1 : (tensor<5x2x3xf32>) -> tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}

// CHECK-LABEL: func.func @reshaper2
// CHECK: tensor.collapse_shape
// CHECK: tensor<5x6xf32>

func.func @reshaper3(%arg1: tensor<5x6xf32>) -> tensor<5x2x3xf32> {
  %0 = tile.reshape %arg1 : (tensor<5x6xf32>) -> tensor<5x2x3xf32>
  return %0 : tensor<5x2x3xf32>
}

// CHECK-LABEL: func.func @reshaper3
// CHECK: tensor.expand_shape
// CHECK: tensor<5x6xf32>

func.func @squeeze(%arg0: tensor<4x2x1x3x2xf32>) -> tensor<4x2x3x2xf32> {
  %0 = tile.reshape %arg0 : (tensor<4x2x1x3x2xf32>) -> tensor<4x2x3x2xf32>
  return %0 : tensor<4x2x3x2xf32>
}

// CHECK-LABEL: func.func @squeeze
// CHECK: tensor.collapse_shape
// CHECK: tensor<4x2x3x2xf32>

func.func @zero_dim(%arg0: tensor<si32>) -> tensor<1x1x1xsi32> {
  %0 = tile.reshape %arg0 : (tensor<si32>) -> tensor<1x1x1xsi32>
  return %0 : tensor<1x1x1xsi32>
}

// CHECK: func.func @zero_dim
// CHECK-SAME: (%[[arg0:.*]]: tensor<i32>) -> tensor<1x1x1xi32>
// CHECK:   tensor.expand_shape %[[arg0]] [] : tensor<i32> into tensor<1x1x1xi32>

func.func @general_reshape(%arg0: tensor<4x3x70x2xf32>) -> tensor<14x10x6x2xf32> {
  %0 = tile.reshape %arg0 : (tensor<4x3x70x2xf32>) -> tensor<14x10x6x2xf32>
  return %0 : tensor<14x10x6x2xf32>
}

// CHECK-LABEL: func.func @general_reshape
// CHECK-SAME: (%[[arg0:.*]]: tensor<4x3x70x2xf32>) -> tensor<14x10x6x2xf32>
// CHECK:   %[[tmp:.*]] = tensor.collapse_shape %[[arg0]]
// CHECK-SAME{LITERAL}: [[0, 1, 2, 3]] : tensor<4x3x70x2xf32> into tensor<1680xf32>
// CHECK:   tensor.expand_shape %[[tmp]]
// CHECK-SAME{LITERAL}: [[0, 1, 2, 3]] : tensor<1680xf32> into tensor<14x10x6x2xf32>

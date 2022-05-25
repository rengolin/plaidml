// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

module {
  func.func @bit_right_scalar(%arg0: tensor<3x3xui64>) -> tensor<3x3xui64> {
    %c9 = tile.constant(9 : i64) : tensor<si32>
    %0 = tile.bit_shr %arg0, %c9 : (tensor<3x3xui64>, tensor<si32>) -> tensor<3x3xui64>
    return %0 : tensor<3x3xui64>
  }
}

// CHECK-LABEL: func.func @bit_right_scalar
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: arith.shrui
// CHECK: pxa.reduce assign

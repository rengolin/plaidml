// Copyright 2022 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

struct ReorderLoopsPass : public ReorderLoopsBase<ReorderLoopsPass> {
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](AffineParallelOp op) { runOnAffineParallel(op); });
  }

  void runOnAffineParallel(AffineParallelOp op) {}
};

} // namespace

std::unique_ptr<Pass> createReorderLoopsPass() {
  return std::make_unique<ReorderLoopsPass>();
}

} // namespace pmlc::dialect::pxa

// Copyright 2020, Intel Corporation

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct MaterializePass : public MaterializeBase<MaterializePass> {
  void runOnOperation() final {
    auto func = getOperation();
    func.walk([&](MaterializeOperandsOpInterface op) {
      OpBuilder builder(op);
      if (failed(op.materializeOperands(builder))) {
        op.emitOpError("Failed to materialize operands");
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createMaterializePass() {
  return std::make_unique<MaterializePass>();
}

} // namespace pmlc::dialect::tile

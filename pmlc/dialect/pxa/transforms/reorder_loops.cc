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

class LoopOrderModel final {
public:
  void setCacheLine(unsigned size) { cacheLine = size; }

  SmallVector<unsigned, 4> evaluate(AffineParallelOp op) {
    SmallVector<unsigned, 4> order;
    for (unsigned i = 0; i < op.getIVs().size(); ++i) {
      order.emplace_back(i);
    }
    return order;
  }

private:
  unsigned cacheLine;
};

struct ReorderLoopsPass : public ReorderLoopsBase<ReorderLoopsPass> {
  explicit ReorderLoopsPass(unsigned cacheLine) {
    loopOrder.setCacheLine(cacheLine);
  }

  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](AffineParallelOp op) {
      if (op.getConstantRanges()) {
        reorder(op, loopOrder.evaluate(op));
      }
    });
  }

  void reorder(AffineParallelOp op, ArrayRef<unsigned> argOrder) {
    auto reductions =
        llvm::to_vector<4>(llvm::map_range(op.reductions(), [](Attribute attr) {
          return attr.cast<AtomicRMWKindAttr>().getValue();
        }));
    auto ranges = *op.getConstantRanges();
    SmallVector<AtomicRMWKind, 4> newReductions;
    SmallVector<int64_t, 4> newRanges;
    for (unsigned pos : argOrder) {
      newReductions.emplace_back(reductions[pos]);
      newRanges.emplace_back(ranges[pos]);
    }

    OpBuilder builder(op->getParentOp());
    auto newOp = builder.create<AffineParallelOp>(
        op.getLoc(), op.getResultTypes(), newReductions, newRanges);
    auto &destOps = newOp.getBody()->getOperations();
    destOps.splice(destOps.begin(), op.getBody()->getOperations());
    auto origArgs = op.getIVs();
    for (auto newArg : newOp.getIVs()) {
      auto pos = argOrder[newArg.getArgNumber()];
      origArgs[pos].replaceAllUsesWith(newArg);
    }
    op.replaceAllUsesWith(newOp);
  }

private:
  LoopOrderModel loopOrder;
};

} // namespace

std::unique_ptr<Pass> createReorderLoopsPass(unsigned cacheLine) {
  return std::make_unique<ReorderLoopsPass>(cacheLine);
}

} // namespace pmlc::dialect::pxa

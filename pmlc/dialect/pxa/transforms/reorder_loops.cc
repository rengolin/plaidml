// Copyright 2022 Intel Corporation

#include <numeric>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/memref_access.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

struct MemRefAccessCounter {
  explicit MemRefAccessCounter(Operation *op) : count(1), access(op) {
    if (auto read = dyn_cast<PxaReadOpInterface>(op)) {
      shape = read.getMemRef().getType().cast<MemRefType>().getShape();
    } else if (auto reduce = dyn_cast<PxaReduceOpInterface>(op)) {
      shape = reduce.getMemRef().getType().cast<MemRefType>().getShape();
    } else {
      op->emitError("Invalid operation for MemRefAccessCounter.");
    }
  }

  bool operator==(const MemRefAccessCounter &rhs) {
    return rhs.shape == shape && rhs.access == access;
  }

  int64_t size() {
    return count * std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<int64_t>());
  }

  SmallVector<unsigned, 4> getBestOrder() {
    SmallVector<unsigned, 4> order;
    return order;
  }

  MemRefAccess access;
  ArrayRef<int64_t> shape;
  unsigned count;
};

class LoopOrderModel final {
public:
  void setCacheLine(unsigned size) { cacheLine = size; }

  SmallVector<unsigned, 4> evaluate(AffineParallelOp op) {
    // Collect the memref access patterns
    SmallVector<MemRefAccessCounter, 4> memrefs;
    op.walk([&](Operation *op) {
      MemRefAccessCounter newAccess(op);
      auto iter = std::find(memrefs.begin(), memrefs.end(), newAccess);
      if (iter == memrefs.end()) {
        memrefs.emplace_back(newAccess);
      } else {
        ++iter->count;
      }
    });

    if (memrefs.empty()) {
      return {};
    }

    // Find out the largest memref access pattern
    unsigned largestIdx = 0;
    int64_t largestSize = memrefs[0].size();
    for (unsigned i = 1; i < memrefs.size(); ++i) {
      int64_t iSize = memrefs[i].size();
      if (iSize > largestSize) {
        largestSize = iSize;
        largestIdx = i;
      }
    }

    return memrefs[largestIdx].getBestOrder();
  }

private:
  unsigned cacheLine;
};

struct ReorderLoopsPass : public ReorderLoopsBase<ReorderLoopsPass> {
  explicit ReorderLoopsPass(unsigned cacheLine, unsigned loopLevels)
      : loopLevels(loopLevels) {
    loopOrder.setCacheLine(cacheLine);
  }

  void collectInnermostLoops() {
    getFunction().walk([&](AffineParallelOp op) {
      int hasInnerLoop = false;
      op.walk([&](AffineParallelOp inner) { hasInnerLoop = true; });
      if (!hasInnerLoop) {
        loopToLevel[op.getOperation()] = 0;
      }
    });
  }

  void setLoopLevels() {
    loopToLevel.clear();
    collectInnermostLoops();
    unsigned level = 1;
    bool updated = true;
    while (updated) {
      updated = false;
      for (auto kvp : loopToLevel) {
        if (kvp.second != level - 1) {
          continue;
        }
        auto curr = kvp.first;
        auto parentOp = curr->getParentOp();
        if (auto parentLoop = dyn_cast<AffineParallelOp>(parentOp)) {
          loopToLevel[parentOp] = level;
          updated = true;
        }
      }
      ++level;
    }
  }

  void runOnFunction() final {
    auto func = getFunction();
    setLoopLevels();
    for (unsigned i = 0; i < loopLevels; ++i) {
      for (auto kvp : loopToLevel) {
        if (kvp.second == i) {
          auto loop = cast<AffineParallelOp>(kvp.first);
          reorder(loop, loopOrder.evaluate(loop));
        }
      }
    }
  }

  void reorder(AffineParallelOp op, ArrayRef<unsigned> argOrder) {
    auto reductions =
        llvm::to_vector<4>(llvm::map_range(op.reductions(), [](Attribute attr) {
          return attr.cast<AtomicRMWKindAttr>().getValue();
        }));
    auto ranges = *op.getConstantRanges();
    SmallVector<int64_t, 4> newRanges;
    for (unsigned pos : argOrder) {
      newRanges.emplace_back(ranges[pos]);
    }

    OpBuilder builder(op->getParentOp());
    builder.setInsertionPoint(op);
    auto newOp = builder.create<AffineParallelOp>(
        op.getLoc(), op.getResultTypes(), reductions, newRanges);
    auto &destOps = newOp.getBody()->getOperations();
    destOps.splice(destOps.begin(), op.getBody()->getOperations());
    auto origArgs = op.getIVs();
    for (auto newArg : newOp.getIVs()) {
      auto pos = argOrder[newArg.getArgNumber()];
      origArgs[pos].replaceAllUsesWith(newArg);
    }
    op.replaceAllUsesWith(newOp);
    op.erase();
  }

private:
  unsigned loopLevels;
  LoopOrderModel loopOrder;
  DenseMap<Operation *, int> loopToLevel;
};

} // namespace

std::unique_ptr<Pass> createReorderLoopsPass(unsigned cacheLine,
                                             unsigned loopLevels) {
  return std::make_unique<ReorderLoopsPass>(cacheLine, loopLevels);
}

} // namespace pmlc::dialect::pxa

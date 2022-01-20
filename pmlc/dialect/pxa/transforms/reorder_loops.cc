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
  explicit MemRefAccessCounter(Operation *op, ArrayRef<BlockArgument> args)
      : access(op), count(1) {
    if (auto read = dyn_cast<PxaReadOpInterface>(op)) {
      type = read.getMemRef().getType().cast<MemRefType>();
    } else if (auto reduce = dyn_cast<PxaReduceOpInterface>(op)) {
      type = reduce.getMemRef().getType().cast<MemRefType>();
    } else {
      op->emitError("Invalid operation for MemRefAccessCounter.");
    }
    for (unsigned i = 0; i < args.size(); ++i) {
      argIdxs[args[i]] = i;
    }
  }

  bool operator==(const MemRefAccessCounter &rhs) {
    return rhs.type == type && rhs.access == access;
  }

  int64_t size() {
    auto shape = type.getShape();
    return count * std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<int64_t>());
  }

  SmallVector<unsigned, 4> getBestOrder() {
    SmallVector<StrideInfo> strideInfo;
    auto si = computeStrideInfo(type, access.accessMap.getAffineMap(),
                                access.accessMap.getOperands());
    if (!si) {
      return {};
    }
    SmallVector<std::pair<BlockArgument, int64_t>> strides;
    for (auto kvp : si->strides) {
      if (argIdxs.find(kvp.first) != argIdxs.end()) {
        strides.emplace_back(kvp.first, kvp.second);
      }
    }
    std::sort(strides.begin(), strides.end(),
              [](const std::pair<BlockArgument, int64_t> &s0,
                 const std::pair<BlockArgument, int64_t> &s1) {
                return s0.second < s1.second;
              });
    SmallVector<unsigned, 4> order;
    DenseSet<BlockArgument> existed;
    for (unsigned i = 0; i < strides.size(); ++i) {
      BlockArgument arg = strides[i].first;
      order.emplace_back(argIdxs[arg]);
      existed.insert(arg);
    }
    for (auto kvp : argIdxs) {
      if (existed.find(kvp.first) == existed.end()) {
        order.emplace_back(kvp.second);
      }
    }
    std::reverse(order.begin(), order.end());
    return order;
  }

  MemRefAccess access;
  DenseMap<BlockArgument, unsigned> argIdxs;
  MemRefType type;
  unsigned count;
};

class LoopOrderModel final {
public:
  void setCacheLine(unsigned size) { cacheLine = size; }

  SmallVector<unsigned, 4> evaluate(AffineParallelOp loop) {
    // Collect the memref access patterns
    SmallVector<MemRefAccessCounter, 4> memrefs;
    loop.walk([&](Operation *op) {
      if (!isa<PxaReadOpInterface>(op) && !isa<PxaReduceOpInterface>(op)) {
        return;
      }
      MemRefAccessCounter newAccess(op, loop.getIVs());
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
      op.walk([&](AffineParallelOp inner) {
        if (inner != op) {
          hasInnerLoop = true;
        }
      });
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
          auto order = loopOrder.evaluate(loop);
          if (!order.empty()) {
            reorder(loop, order);
          }
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
    newOp->setAttrs(op->getAttrs());
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

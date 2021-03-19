// Copyright 2020 Intel Corporation

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/util/logging.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

/// Walks over all immediate return-like terminators in the given region.
template <typename FuncT>
static void walkReturnOperations(Region *region, const FuncT &func) {
  for (Block &block : *region)
    for (Operation &operation : block) {
      // Skip non-return-like terminators.
      if (operation.hasTrait<OpTrait::ReturnLike>())
        func(&operation);
    }
}

namespace {

//===----------------------------------------------------------------------===//
// Backedges analysis
//===----------------------------------------------------------------------===//

/// A straight-forward program analysis which detects loop backedges induced by
/// explicit control flow.
class Backedges {
public:
  using BlockSetT = SmallPtrSet<Block *, 16>;
  using BackedgeSetT = llvm::DenseSet<std::pair<Block *, Block *>>;

public:
  /// Constructs a new backedges analysis using the op provided.
  explicit Backedges(Operation *op) { recurse(op, op->getBlock()); }

  /// Returns the number of backedges formed by explicit control flow.
  size_t size() const { return edgeSet.size(); }

  /// Returns the start iterator to loop over all backedges.
  BackedgeSetT::const_iterator begin() const { return edgeSet.begin(); }

  /// Returns the end iterator to loop over all backedges.
  BackedgeSetT::const_iterator end() const { return edgeSet.end(); }

private:
  /// Enters the current block and inserts a backedge into the `edgeSet` if we
  /// have already visited the current block. The inserted edge links the given
  /// `predecessor` with the `current` block.
  bool enter(Block &current, Block *predecessor) {
    bool inserted = visited.insert(&current).second;
    if (!inserted)
      edgeSet.insert(std::make_pair(predecessor, &current));
    return inserted;
  }

  /// Leaves the current block.
  void exit(Block &current) { visited.erase(&current); }

  /// Recurses into the given operation while taking all attached regions into
  /// account.
  void recurse(Operation *op, Block *predecessor) {
    Block *current = op->getBlock();
    // If the current op implements the `BranchOpInterface`, there can be
    // cycles in the scope of all successor blocks.
    if (isa<BranchOpInterface>(op)) {
      for (Block *succ : current->getSuccessors())
        recurse(*succ, current);
    }
    // Recurse into all distinct regions and check for explicit control-flow
    // loops.
    for (Region &region : op->getRegions())
      recurse(region.front(), current);
  }

  /// Recurses into explicit control-flow structures that are given by
  /// the successor relation defined on the block level.
  void recurse(Block &block, Block *predecessor) {
    // Try to enter the current block. If this is not possible, we are
    // currently processing this block and can safely return here.
    if (!enter(block, predecessor))
      return;

    // Recurse into all operations and successor blocks.
    for (Operation &op : block.getOperations())
      recurse(&op, predecessor);

    // Leave the current block.
    exit(block);
  }

  /// Stores all blocks that are currently visited and on the processing stack.
  BlockSetT visited;

  /// Stores all backedges in the format (source, target).
  BackedgeSetT edgeSet;
};

//===----------------------------------------------------------------------===//
// BufferDeallocation
//===----------------------------------------------------------------------===//

/// The buffer deallocation transformation which ensures that all allocs in the
/// program have a corresponding de-allocation. As a side-effect, it might also
/// introduce copies that in turn leads to additional allocs and de-allocations.
class BufferDeallocation : BufferPlacementTransformationBase {
public:
  explicit BufferDeallocation(Operation *op)
      : BufferPlacementTransformationBase(op), dominators(op),
        postDominators(op) {}

  /// Performs the actual placement/creation of all temporary alloc, copy and
  /// dealloc nodes.
  void deallocate() {
    // Add additional allocations and copies that are required.
    introduceCopies();
    // Place deallocations for all allocation entries.
    placeDeallocs();
  }

private:
  /// Introduces required allocs and copy operations to avoid memory leaks.
  void introduceCopies() {
    // Initialize the set of values that require a dedicated memory free
    // operation since their operands cannot be safely deallocated in a post
    // dominator.
    SmallPtrSet<Value, 8> valuesToFree;
    llvm::SmallDenseSet<std::tuple<Value, Block *>> visitedValues;
    SmallVector<std::tuple<Value, Block *>, 8> toProcess;

    // Check dominance relation for proper dominance properties. If the given
    // value node does not dominate an alias, we will have to create a copy in
    // order to free all buffers that can potentially leak into a post
    // dominator.
    auto findUnsafeValues = [&](Value source, Block *definingBlock) {
      auto it = aliases.find(source);
      if (it == aliases.end())
        return;
      for (Value value : it->second) {
        if (valuesToFree.count(value) > 0)
          continue;
        Block *parentBlock = value.getParentBlock();
        // Check whether we have to free this particular block argument or
        // generic value. We have to free the current alias if it is either
        // defined in a non-dominated block or it is defined in the same block
        // but the current value is not dominated by the source value.
        if (!dominators.dominates(definingBlock, parentBlock) ||
            (definingBlock == parentBlock && value.isa<BlockArgument>())) {
          toProcess.emplace_back(value, parentBlock);
          valuesToFree.insert(value);
        } else if (visitedValues.insert(std::make_tuple(value, definingBlock))
                       .second) {
          toProcess.emplace_back(value, definingBlock);
        }
      }
    };

    // Detect possibly unsafe aliases starting from all allocations.
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);
      findUnsafeValues(allocValue, allocValue.getDefiningOp()->getBlock());
    }
    // Try to find block arguments that require an explicit free operation
    // until we reach a fix point.
    while (!toProcess.empty()) {
      auto current = toProcess.pop_back_val();
      findUnsafeValues(std::get<0>(current), std::get<1>(current));
    }

    // Update buffer aliases to ensure that we free all buffers and block
    // arguments at the correct locations.
    aliases.remove(valuesToFree);

    // Add new allocs and additional copy operations.
    for (Value value : valuesToFree) {
      if (auto blockArg = value.dyn_cast<BlockArgument>())
        introduceBlockArgCopy(blockArg);
      else
        introduceValueCopyForRegionResult(value);

      // Register the value to require a final dealloc. Note that we do not have
      // to assign a block here since we do not want to move the allocation node
      // to another location.
      allocs.registerAlloc(std::make_tuple(value, nullptr));
    }
  }

  /// Introduces temporary allocs in all predecessors and copies the source
  /// values into the newly allocated buffers.
  void introduceBlockArgCopy(BlockArgument blockArg) {
    // Allocate a buffer for the current block argument in the block of
    // the associated value (which will be a predecessor block by
    // definition).
    Block *block = blockArg.getOwner();
    for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
      // Get the terminator and the value that will be passed to our
      // argument.
      Operation *terminator = (*it)->getTerminator();
      auto branchInterface = cast<BranchOpInterface>(terminator);
      // Query the associated source value.
      Value sourceValue =
          branchInterface.getSuccessorOperands(it.getSuccessorIndex())
              .getValue()[blockArg.getArgNumber()];
      // Create a new alloc and copy at the current location of the terminator.
      Value alloc = introduceBufferCopy(sourceValue, terminator);
      // Wire new alloc and successor operand.
      auto mutableOperands =
          branchInterface.getMutableSuccessorOperands(it.getSuccessorIndex());
      if (!mutableOperands.hasValue())
        terminator->emitError() << "terminators with immutable successor "
                                   "operands are not supported";
      else
        mutableOperands.getValue()
            .slice(blockArg.getArgNumber(), 1)
            .assign(alloc);
    }

    // Check whether the block argument has implicitly defined predecessors via
    // the RegionBranchOpInterface. This can be the case if the current block
    // argument belongs to the first block in a region and the parent operation
    // implements the RegionBranchOpInterface.
    Region *argRegion = block->getParent();
    Operation *parentOp = argRegion->getParentOp();
    RegionBranchOpInterface regionInterface;
    if (!argRegion || &argRegion->front() != block ||
        !(regionInterface = dyn_cast<RegionBranchOpInterface>(parentOp)))
      return;

    introduceCopiesForRegionSuccessors(
        regionInterface, argRegion->getParentOp()->getRegions(), blockArg,
        [&](RegionSuccessor &successorRegion) {
          // Find a predecessor of our argRegion.
          return successorRegion.getSuccessor() == argRegion;
        });

    // Check whether the block argument belongs to an entry region of the
    // parent operation. In this case, we have to introduce an additional copy
    // for buffer that is passed to the argument.
    SmallVector<RegionSuccessor, 2> successorRegions;
    regionInterface.getSuccessorRegions(/*index=*/llvm::None, successorRegions);
    auto *it =
        llvm::find_if(successorRegions, [&](RegionSuccessor &successorRegion) {
          return successorRegion.getSuccessor() == argRegion;
        });
    if (it == successorRegions.end())
      return;

    // Determine the actual operand to introduce a copy for and rewire the
    // operand to point to the copy instead.
    Value operand =
        regionInterface.getSuccessorEntryOperands(argRegion->getRegionNumber())
            [llvm::find(it->getSuccessorInputs(), blockArg).getIndex()];
    Value copy = introduceBufferCopy(operand, parentOp);

    auto op = llvm::find(parentOp->getOperands(), operand);
    assert(op != parentOp->getOperands().end() &&
           "parentOp does not contain operand");
    parentOp->setOperand(op.getIndex(), copy);
  }

  /// Introduces temporary allocs in front of all associated nested-region
  /// terminators and copies the source values into the newly allocated buffers.
  void introduceValueCopyForRegionResult(Value value) {
    // Get the actual result index in the scope of the parent terminator.
    Operation *operation = value.getDefiningOp();
    auto regionInterface = cast<RegionBranchOpInterface>(operation);
    // Filter successors that return to the parent operation.
    auto regionPredicate = [&](RegionSuccessor &successorRegion) {
      // If the RegionSuccessor has no associated successor, it will return to
      // its parent operation.
      return !successorRegion.getSuccessor();
    };
    // Introduce a copy for all region "results" that are returned to the parent
    // operation. This is required since the parent's result value has been
    // considered critical. Therefore, the algorithm assumes that a copy of a
    // previously allocated buffer is returned by the operation (like in the
    // case of a block argument).
    introduceCopiesForRegionSuccessors(regionInterface, operation->getRegions(),
                                       value, regionPredicate);
  }

  /// Introduces buffer copies for all terminators in the given regions. The
  /// regionPredicate is applied to every successor region in order to restrict
  /// the copies to specific regions.
  template <typename TPredicate>
  void introduceCopiesForRegionSuccessors(
      RegionBranchOpInterface regionInterface, MutableArrayRef<Region> regions,
      Value argValue, const TPredicate &regionPredicate) {
    for (Region &region : regions) {
      // Query the regionInterface to get all successor regions of the current
      // one.
      SmallVector<RegionSuccessor, 2> successorRegions;
      regionInterface.getSuccessorRegions(region.getRegionNumber(),
                                          successorRegions);
      // Try to find a matching region successor.
      RegionSuccessor *regionSuccessor =
          llvm::find_if(successorRegions, regionPredicate);
      if (regionSuccessor == successorRegions.end())
        continue;
      // Get the operand index in the context of the current successor input
      // bindings.
      size_t operandIndex =
          llvm::find(regionSuccessor->getSuccessorInputs(), argValue)
              .getIndex();

      // Iterate over all immediate terminator operations to introduce
      // new buffer allocations. Thereby, the appropriate terminator operand
      // will be adjusted to point to the newly allocated buffer instead.
      walkReturnOperations(&region, [&](Operation *terminator) {
        // Extract the source value from the current terminator.
        Value sourceValue = terminator->getOperand(operandIndex);
        // Create a new alloc at the current location of the terminator.
        Value alloc = introduceBufferCopy(sourceValue, terminator);
        // Wire alloc and terminator operand.
        terminator->setOperand(operandIndex, alloc);
      });
    }
  }

  /// Creates a new memory allocation for the given source value and copies
  /// its content into the newly allocated buffer. The terminator operation is
  /// used to insert the alloc and copy operations at the right places.
  Value introduceBufferCopy(Value sourceValue, Operation *terminator) {
    // Avoid multiple copies of the same source value. This can happen in the
    // presence of loops when a branch acts as a backedge while also having
    // another successor that returns to its parent operation. Note: that
    // copying copied buffers can introduce memory leaks since the invariant of
    // BufferPlacement assumes that a buffer will be only copied once into a
    // temporary buffer. Hence, the construction of copy chains introduces
    // additional allocations that are not tracked automatically by the
    // algorithm.
    if (copiedValues.contains(sourceValue))
      return sourceValue;
    // Create a new alloc at the current location of the terminator.
    auto memRefType = sourceValue.getType().cast<MemRefType>();
    OpBuilder builder(terminator);

    // Extract information about dynamically shaped types by
    // extracting their dynamic dimensions.
    auto dynamicOperands =
        getDynOperands(terminator->getLoc(), sourceValue, builder);

    // TODO: provide a generic interface to create dialect-specific
    // Alloc and CopyOp nodes.
    auto alloc = builder.create<AllocOp>(terminator->getLoc(), memRefType,
                                         dynamicOperands);

    // Create a new copy operation that copies to contents of the old
    // allocation to the new one.
    builder.create<linalg::CopyOp>(terminator->getLoc(), sourceValue, alloc);

    // Remember the copy of original source value.
    copiedValues.insert(alloc);
    return alloc;
  }

  /// Finds correct dealloc positions according to the algorithm described at
  /// the top of the file for all alloc nodes and block arguments that can be
  /// handled by this analysis.
  void placeDeallocs() const {
    // Move or insert deallocs using the previously computed information.
    // These deallocations will be linked to their associated allocation nodes
    // since they don't have any aliases that can (potentially) increase their
    // liveness.
    for (const BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value alloc = std::get<0>(entry);
      auto aliasesSet = aliases.resolve(alloc);
      assert(aliasesSet.size() > 0 && "must contain at least one alias");

      // Determine the actual block to place the dealloc and get liveness
      // information.
      Block *placementBlock =
          findCommonDominator(alloc, aliasesSet, postDominators);
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(placementBlock);

      // We have to ensure that the dealloc will be after the last use of all
      // aliases of the given value. We first assume that there are no uses in
      // the placementBlock and that we can safely place the dealloc at the
      // beginning.
      Operation *endOperation = &placementBlock->front();

      // Iterate over all aliases and ensure that the endOperation will point
      // to the last operation of all potential aliases in the placementBlock.
      for (Value alias : aliasesSet) {
        // Ensure that the start operation is at least the defining operation of
        // the current alias to avoid invalid placement of deallocs for aliases
        // without any uses.
        Operation *beforeOp = endOperation;
        if (alias.getDefiningOp() &&
            !(beforeOp = placementBlock->findAncestorOpInBlock(
                  *alias.getDefiningOp())))
          continue;

        Operation *aliasEndOperation =
            livenessInfo->getEndOperation(alias, beforeOp);
        // Check whether the aliasEndOperation lies in the desired block and
        // whether it is behind the current endOperation. If yes, this will be
        // the new endOperation.
        if (aliasEndOperation->getBlock() == placementBlock &&
            endOperation->isBeforeInBlock(aliasEndOperation))
          endOperation = aliasEndOperation;
      }
      // endOperation is the last operation behind which we can safely store
      // the dealloc taking all potential aliases into account.

      // If there is an existing dealloc, move it to the right place.
      Operation *deallocOperation = std::get<1>(entry);
      if (deallocOperation) {
        deallocOperation->moveAfter(endOperation);
      } else {
        // If the Dealloc position is at the terminator operation of the
        // block, then the value should escape from a deallocation.
        Operation *nextOp = endOperation->getNextNode();
        if (!nextOp)
          continue;
        // If there is no dealloc node, insert one in the right place.
        OpBuilder builder(nextOp);
        builder.create<DeallocOp>(alloc.getLoc(), alloc);
      }
    }
  }

  /// The dominator info to find the appropriate start operation to move the
  /// allocs.
  DominanceInfo dominators;

  /// The post dominator info to move the dependent allocs in the right
  /// position.
  PostDominanceInfo postDominators;

  /// Stores already copied allocations to avoid additional copies of copies.
  ValueSetT copiedValues;
};

} // end anonymous namespace

void runOnMainFunction(FuncOp fn) {
  // Ensure that there are supported loops only.
  Backedges backedges(fn.getOperation());
  if (backedges.size()) {
    fn.emitError("Structured control-flow loops are supported only.");
    return;
  }

  // Place all required temporary alloc, copy and dealloc nodes.
  BufferDeallocation deallocation(fn.getOperation());
  deallocation.deallocate();
}

struct CustomBufferDeallocationPass
    : public CustomBufferDeallocationBase<CustomBufferDeallocationPass> {
  void runOnOperation() final {
    // Get the module
    ModuleOp op = getOperation();
    // Run all functions.  This could almost be a function pass, but init + fini
    // interact, which breaks the independence requirements
    op.walk([&](FuncOp fn) {
      if (fn.getName() == "init") {
        // If the function is named init, find fini
        auto finiFunc = op.lookupSymbol<FuncOp>("fini");
        if (!finiFunc) {
          fn.emitError() << "Init with no fini";
          signalPassFailure();
          return;
        }
        // Find fini's unpack op
        auto unpackOp =
            dyn_cast<pmlc::dialect::stdx::UnpackOp>(finiFunc.begin()->begin());
        if (!unpackOp) {
          finiFunc.emitError() << "Fini must begin with unpack";
          signalPassFailure();
          return;
        }
        // Now, place deallocs on the init functions, moving escaping allocs to
        // be dealloced in fini
        OpBuilder deallocBuilder(finiFunc.begin()->getTerminator());
        runOnFunction(fn, [&](unsigned i) {
          deallocBuilder.create<DeallocOp>(fn.getLoc(), unpackOp.getResult(i));
        });
      } else if (fn.getName() == "fini") {
        // Place allocs, if any escape, it's an error
        runOnFunction(
            fn, [&](unsigned i) {
              fn.emitError()
                  << "Allocations escape via a pack for non-init function";
              signalPassFailure();
            });
      } else {
        runOnMainFunction(fn);
      }
    });
  }

  template <typename Callback>
  void runOnFunction(FuncOp fn, Callback onPack) {
    // Place deallocation for AllocOp
    fn.walk([&](AllocOp alloc) {
      IVLOG(3, "alloc: " << debugString(*alloc));
      placeDealloc(alloc.getResult(), alloc, alloc.getOperation()->getBlock(),
                   onPack);
    });
  }

  // This function dealloc ref if possible, which is the allocated memory
  // reference. firstOp is generally the allocation operaion. For scf.for
  // arguments, it is virtually the first operation in the loop. argNumber is
  // scf.for argument order number, which is useless for normal deallocation.
  template <typename Callback>
  void placeDealloc(Value ref, Operation *firstOp, Block *allocBlock,
                    Callback onPack) {
    Operation *lastOp = firstOp;
    OpOperand *lastUse = nullptr;
    for (auto &itUse : pmlc::dialect::pxa::getIndirectUses(ref)) {
      auto use = itUse.getOwner();
      IVLOG(3, "  use: " << debugString(*use));

      auto ancestor = allocBlock->findAncestorOpInBlock(*use);
      assert(ancestor && "use and alloc do not have a common ancestor");
      IVLOG(3, "  ancestor: " << debugString(*use));

      if (isa<ReturnOp>(ancestor)) {
        IVLOG(3, "  return");
        return;
      }

      if (!ancestor->isBeforeInBlock(lastOp)) {
        lastOp = ancestor;
        lastUse = &itUse;
      }
    }
    IVLOG(3, "  last ancestor: " << debugString(*lastOp));
    if (auto packOp = dyn_cast<pmlc::dialect::stdx::PackOp>(lastOp)) {
      IVLOG(3, "  pack op");
      // Alloc 'escapes' via a pack, call our callback to handle
      onPack(lastUse->getOperandNumber());
      return;
    }

    Operation *nextOp = lastOp->getNextNode();
    if (!nextOp) {
      IVLOG(3, "  terminator");
      return;
    }

    IVLOG(3, "  next operation: " << debugString(*nextOp));
    OpBuilder builder(nextOp);
    builder.create<DeallocOp>(firstOp->getLoc(), lastUse->get());
  }
};

std::unique_ptr<mlir::Pass> createCustomBufferDeallocationPass() {
  return std::make_unique<CustomBufferDeallocationPass>();
}

} // namespace pmlc::target::x86

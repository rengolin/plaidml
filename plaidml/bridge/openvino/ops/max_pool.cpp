// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset8.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

template <typename MaxPoolType>
edsl::Tensor registerMaxPoolOp(MaxPoolType* layer, const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  std::vector<int> strides;
  for (auto stride : layer->get_strides()) {
    strides.push_back(stride);
  }
  std::vector<int> kernel;
  for (auto k : layer->get_kernel()) {
    kernel.push_back(k);
  }
  auto pool_type = plaidml::op::PoolMode::MAX;
  auto input_layout = plaidml::op::TensorLayout::NCX;
  auto autopad_mode = to_plaidml(layer->get_auto_pad());
  bool include_padding_in_avg = false;
  auto use_ceil_for_output_shape = layer->get_rounding_type() == ngraph::op::RoundingType::CEIL;
  std::vector<int> padding;
  if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
    for (auto pad : layer->get_pads_begin()) {
      padding.push_back(pad);
    }
    for (auto pad : layer->get_pads_end()) {
      padding.push_back(pad);
    }
  }
  return op::pool(I, pool_type, kernel, strides, autopad_mode, padding, input_layout, include_padding_in_avg,
                  use_ceil_for_output_shape);
}

void registerMaxPool() {
  registerOp("MaxPool", [](const Context& ctx) {
    if (auto* layer = ngraph::as_type<ngraph::opset1::MaxPool>(ctx.layer)) {
      return edsl::make_tuple(registerMaxPoolOp<ngraph::opset1::MaxPool>(layer, ctx));
    }
    if (auto* layer = ngraph::as_type<ngraph::opset8::MaxPool>(ctx.layer)) {
      // TODO: the second output tensor is useful
      return edsl::make_tuple(registerMaxPoolOp<ngraph::opset8::MaxPool>(layer, ctx), edsl::Tensor{0});
    }
    IE_ASSERT(false);
  });
}

}  // namespace PlaidMLPlugin

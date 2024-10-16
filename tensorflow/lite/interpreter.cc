/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/interpreter.h"

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/delegates/status.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

// TODO(b/139446230): Move to portable platform header.
#if defined(__ANDROID__)
#define TFLITE_IS_MOBILE_PLATFORM
#endif  // defined(__ANDROID__)

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
#define TFLITE_IS_MOBILE_PLATFORM
#elif TARGET_OS_IPHONE
#define TFLITE_IS_MOBILE_PLATFORM
#endif
#endif  // defined(__APPLE__)

// TODO(b/132087118): move static_assert to c_api_internal when compiled with
// C++.
static_assert(sizeof(TfLiteFloat16) == sizeof(uint16_t),
              "Float 16 type must be 16 bits.");

namespace tflite {

namespace {

// Gets the current TfLiteQuantization from the legacy TfLiteQuantizationParams.
TfLiteQuantization GetQuantizationFromLegacy(
    const TfLiteQuantizationParams& legacy_quantization) {
  TfLiteQuantization quantization;
  quantization.type = kTfLiteAffineQuantization;
  auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  affine_quantization->scale = TfLiteFloatArrayCreate(1);
  affine_quantization->zero_point = TfLiteIntArrayCreate(1);
  affine_quantization->scale->data[0] = legacy_quantization.scale;
  affine_quantization->zero_point->data[0] = legacy_quantization.zero_point;
  quantization.params = affine_quantization;

  return quantization;
}

// TODO(b/153131797): We have put 'delegate_status' to 0 in the following macro
// temporarily because delegate-specific error codes are either not retrievable
// at the moment, which we will add later.
#define TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(runtime_event, a) \
  do {                                                                      \
    TfLiteStatus status = (a);                                              \
    runtime_event.set_runtime_status(/*delegate_status=*/0,                 \
                                     static_cast<int64_t>(status));         \
    TF_LITE_ENSURE_STATUS(status);                                          \
  } while (0)

}  // namespace

Interpreter::Interpreter(ErrorReporter* error_reporter)
    : error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {
  // TODO(b/128420794): Include the TFLite runtime version in the log.
  // Prod logging is useful for mobile platforms where scraping console logs is
  // critical for debugging.

  std::cout << "Interpreter : Initializing default tflite interpreter"
            << "\n";
#if defined(TFLITE_IS_MOBILE_PLATFORM)
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#else
  TFLITE_LOG_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#endif

  // legacy
  // // There's always at least 1 subgraph which is the primary subgraph.
  AddSubgraphs(1);
  context_ = primary_subgraph().context();

  // Reserve some space for the tensors to avoid excessive resizing.
  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    external_contexts_[i] = nullptr;
  }

  // This operation is cheap because we allocate the CPU context resources (i.e.
  // threads) lazily.
  own_external_cpu_backend_context_.reset(new ExternalCpuBackendContext());
  external_contexts_[kTfLiteCpuBackendContext] =
      own_external_cpu_backend_context_.get();

  // legacy
  primary_subgraph().UseNNAPI(false);


  // THIS CODE IS DEPRECATED
  // scheduler_ = new LiteScheduler(this); //create scheduler thread here
  // std::cout << "Interperter Created with new job queue and scheduler" <<
  // "\n";
}

Interpreter::Interpreter(bool use_job) {
  error_reporter_ = DefaultErrorReporter();
  // TODO(b/128420794): Include the TFLite runtime version in the log.
  // Prod logging is useful for mobile platforms where scraping console logs is
  // critical for debugging.

  std::cout << "Interpreter : Initializing modified tflite interpreter"
            << "\n";
#if defined(TFLITE_IS_MOBILE_PLATFORM)
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#else
  TFLITE_LOG_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#endif

  // legacy
  // // There's always at least 1 subgraph which is the primary subgraph.
  // AddSubgraphs(1);
  // context_ = primary_subgraph().context();

  // Reserve some space for the tensors to avoid excessive resizing.
  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    external_contexts_[i] = nullptr;
  }

  // This operation is cheap because we allocate the CPU context resources (i.e.
  // threads) lazily.
  own_external_cpu_backend_context_.reset(new ExternalCpuBackendContext());
  external_contexts_[kTfLiteCpuBackendContext] =
      own_external_cpu_backend_context_.get();

  // legacy
  // primary_subgraph().UseNNAPI(false);
}

Interpreter::~Interpreter() {
  // The owned external Cpu Backend Context will go out of scope with this
  // interpreter. If we have an external backend context that is not
  // owned, we need to clear the cache for other interpreters that may
  // use the context.
  if (external_contexts_[kTfLiteCpuBackendContext] &&
      (external_contexts_[kTfLiteCpuBackendContext] !=
       own_external_cpu_backend_context_.get())) {
    ExternalCpuBackendContext* external_context =
        static_cast<ExternalCpuBackendContext*>(
            external_contexts_[kTfLiteCpuBackendContext]);
    TfLiteInternalBackendContext* internal_context =
        external_context->internal_backend_context();
    if (internal_context) {
      // This call may have negative performance impacts on the next inference
      // for any interpreter using this context. The cache will be refreshed
      // by the next inference.
      internal_context->ClearCaches();
    }
  }
  std::cout << "Interpreter destructor called"
            << "\n";
}

void Interpreter::SetExternalContext(TfLiteExternalContextType type,
                                     TfLiteExternalContext* ctx) {
  if (ctx == own_external_cpu_backend_context_.get()) {
    error_reporter_->Report(
        "WARNING: The passed external context is identical to the internally "
        "owned one.");
    return;
  }

  // We have an internally owned external context of kTfLiteCpuBackendContext.
  // If it's overwritten here, we will release the resource of the internally
  // owned external context.
  // Note: the 'max thread count' info associated with the overwritten context
  // will be lost here, and such info is now determined by the new context, thus
  // affecting how much parallelism a TFLite op would have.
  if (kTfLiteCpuBackendContext == type &&
      external_contexts_[kTfLiteCpuBackendContext] ==
          own_external_cpu_backend_context_.get()) {
    own_external_cpu_backend_context_.reset();
  }

  // This essentially changes the "external_contexts_[type]".
  primary_subgraph().SetExternalContext(type, ctx);
}

TfLiteStatus Interpreter::SetCustomAllocationForTensor(
    int tensor_index, const TfLiteCustomAllocation& allocation) {
  return primary_subgraph().SetCustomAllocationForTensor(tensor_index,
                                                         allocation);
}

TfLiteStatus Interpreter::SetInputs(std::vector<int> inputs) {
  return primary_subgraph().SetInputs(std::move(inputs));
}

TfLiteStatus Interpreter::SetOutputs(std::vector<int> outputs) {
  return primary_subgraph().SetOutputs(std::move(outputs));
}

TfLiteStatus Interpreter::SetVariables(std::vector<int> variables) {
  return primary_subgraph().SetVariables(std::move(variables));
}

TfLiteStatus Interpreter::AllocateTensors() {
  // Apply the default delegate that TFLite will enable at this point to allow
  // other user-level delegates to be applied first.
  if (!lazy_delegate_providers_.empty()) {
    TFLITE_LOG(TFLITE_LOG_INFO,
               "Applying %zu TensorFlow Lite delegate(s) lazily.",
               lazy_delegate_providers_.size());
    // At the momement, XNNPACK delegate is the only one that might be applied
    // by default, in which case, the execution will fall back to default
    // implementation if the XNNPACK delegate fails to be applied. Therefore, we
    // ignore the return status here and let it fall through the rest of the
    // code.
    for (size_t i = 0; i < lazy_delegate_providers_.size(); ++i) {
      auto status =
          ModifyGraphWithDelegate(std::move(lazy_delegate_providers_[i]));
      switch (status) {
        case kTfLiteOk:
          TFLITE_LOG(TFLITE_LOG_INFO,
                     "Successfully applied the default TensorFlow Lite "
                     "delegate indexed at %zu.",
                     i);
          break;
        case kTfLiteError:
          TF_LITE_REPORT_ERROR(error_reporter_,
                               "Failed to apply the default TensorFlow Lite "
                               "delegate indexed at %zu.",
                               i);
          return kTfLiteError;
        case kTfLiteDelegateError:
          TF_LITE_REPORT_ERROR(
              error_reporter_,
              "Error in applying the default TensorFlow Lite delegate indexed "
              "at %zu, and all previously applied delegates are reverted.",
              i);
          break;
        case kTfLiteApplicationError:
          TF_LITE_REPORT_ERROR(error_reporter_,
                               "Ignoring failed application of the default "
                               "TensorFlow Lite delegate indexed at %zu.",
                               i);
          break;
        default:
          TF_LITE_REPORT_ERROR(error_reporter_,
                               "Unknown status (%d) after applying the default "
                               "TensorFlow Lite delegate indexed at %zu.",
                               status, i);
          return kTfLiteError;
      }
    }
    lazy_delegate_providers_.clear();
  }

  return primary_subgraph().AllocateTensors();
}

// Minsung
// First, partition subgraphs in height-wise if needed.
// (because TfLite automatically propagates hw-partitioned tensor dims,
//  we partition hw before allocation.)
// Second, allocate first subgraph of subgraph subset(which have same model id).
// (first subgraph means the subgraph which owns the input tensor of a model)
// Check the input tensor range from it.
// Resize intermediate sharing tensors of all subgraph subset.
// Then, partition other subgraphs in height-wise.
// Finally allocate tensors of all subgraph subset.
// +++ All shared tensors should be input tensors of it's subgraph.
// TfLiteStatus Interpreter::AllocateTensorsofSubsets(int model_id) {
//   auto HeightPartitionandAllocateIfNeed = [&](Subgraph* subgraph) {
//     if (subgraph->GetResourceType() == ResourceType::CO_CPU ||
//         subgraph->GetResourceType() == ResourceType::CO_CPU_XNN ||
//         subgraph->GetResourceType() == ResourceType::CO_GPU) {
//       if (subgraph->GetPartitioningType() ==
//           PartitioningType::HEIGHT_PARTITIONING) {
//         std::cout << "Height partition on subgraph " << subgraph->GetGraphid()
//                   << "\n";
//         if (subgraph->PartitionHeightTest() != kTfLiteOk) {
//           std::cout << "Height partitioning TEST returned ERROR"
//                     << "\n";
//           return kTfLiteError;
//         }
//         if (subgraph->AllocateTensors() != kTfLiteOk) {
//           std::cout << "AllocateTensors after HeightPartitioning returned ERROR"
//                     << "\n";
//           return kTfLiteError;
//         }
//         if (subgraph->AllocateConcateTensors() != kTfLiteOk) {
//           std::cout << "AllocateConcateTensors after HeightPartitioning "
//                        "returned ERROR"
//                     << "\n";
//           return kTfLiteError;
//         }
//       }
//     }
//     return kTfLiteOk;
//   };
//   Subgraph* primary_working_subgraph;
//   for (auto subset : subgraph_subsets) {
//     if (subset.first == model_id) {
//       if (subset.second.size() > 0) {
//         primary_working_subgraph =
//             subgraph_id(subset.second[0]);  // Allocate first subgraph
//         if (primary_working_subgraph->AllocateTensors() != kTfLiteOk) {
//           std::cout << "AllocateTensors of graph [" << subset.second[0] << "] "
//                     << "returned ERROR"
//                     << "\n";
//           return kTfLiteError;
//         }
//         // Resize intermediate shared tensors with GetIntermediateTensorRange()
//         int input_tensor_begin_idx, input_tensor_end_idx;
//         if (GetIntermediateTensorRangeWithGraphSubset(
//                 model_id, &input_tensor_begin_idx, &input_tensor_end_idx) !=
//             kTfLiteOk) {
//           std::cout << "GetIntermediateTensorRangeWithGraphSubset ERROR"
//                     << "\n";
//           return kTfLiteError;
//         }
//         // // Debug print
//         // for(int k=0; k<shared_tensor_and_graph.size(); ++k){
//         //   if(shared_tensor_and_graph[k]->model_id == model_id){
//         //     for(int i=0;
//         //     i<shared_tensor_and_graph[k]->pair_tensor_graph.size(); ++i){
//         //       std::cout << "tensor : " <<
//         //       shared_tensor_and_graph[k]->pair_tensor_graph[i].first << "\n";
//         //       std::cout << "subgraphs : ";
//         //       for(int j=0;
//         //       j<shared_tensor_and_graph[k]->pair_tensor_graph[i].second.size();
//         //       ++j){
//         //         std::cout <<
//         //         shared_tensor_and_graph[k]->pair_tensor_graph[i].second[j] <<
//         //         " ";
//         //       }
//         //       std::cout << "\n";
//         //     }
//         //   }
//         // }
        
//         // Description
//         // First, propagate primiray subgraphs output dims to other subgraphs.
//         // than allocate tensors.
//         for (auto shared_tensor_and_graph_ : shared_tensor_and_graph) {
//           if (shared_tensor_and_graph_->model_id == model_id) {
//             for (int i = 0;
//                  i < shared_tensor_and_graph_->pair_tensor_graph.size(); ++i) {
//               int base_tensor =
//                   shared_tensor_and_graph_->pair_tensor_graph[i].first;
//               if (base_tensor >= input_tensor_begin_idx &&
//                   base_tensor <= input_tensor_end_idx) {
//                 TfLiteTensor* working_tensor;
//                 std::vector<int> match_dims;
//                 for (int j = 0;
//                      j < shared_tensor_and_graph_->pair_tensor_graph[i]
//                              .second.size();
//                      ++j) {
//                   int working_subgraph =
//                       shared_tensor_and_graph_->pair_tensor_graph[i].second[j];
//                   if (j == 0) {
//                     subgraph_id(working_subgraph)->PushToOutputs(base_tensor);
//                     working_tensor =
//                         subgraph_id(working_subgraph)->tensor(base_tensor);
//                     match_dims = subgraph_id(working_subgraph)
//                                      ->GetTensorShape(base_tensor);
//                   } else {
//                     // MUST FIX. REDUNDUNT INPUT, OUTPUT TENSOR!!
//                     // MUST FIX. REDUNDUNT INPUT, OUTPUT TENSOR!!
//                     subgraph_id(working_subgraph)
//                         ->ResizeInputTensor(base_tensor, match_dims);
//                     subgraph_id(working_subgraph)->PushToInputs(base_tensor);
//                   }
//                   if (subgraph_id(working_subgraph)->AllocateTensors() !=
//                       kTfLiteOk)
//                     return kTfLiteError;
//                 }
//                 working_tensor = nullptr;
//                 match_dims.clear();
//               }
//             }
//           }
//         }
//       } else {
//         std::cout << "Interpreter : no registerd subgraph of "
//                   << "model id [" << model_id << "] no allocation occurs.\n";
//         return kTfLiteOk;
//       }
//     }
//     // Second, partition subgraphs and allocate again.
//     for (int subgraph_idx = 0; subgraph_idx < subset.second.size();
//          ++subgraph_idx) {
//       int working_subgraph_id = subset.second[subgraph_idx];
//       Subgraph* working_subgraph = subgraph_id(working_subgraph_id);
//       if (HeightPartitionandAllocateIfNeed(working_subgraph) != kTfLiteOk) {
//         std::cout << "HeightPartitionAndAllocateIfNeed returned ERROR"
//                   << "\n";
//         return kTfLiteError;
//       }
//       // Minsung
//       if (working_subgraph->AllocateConcateTensors() != kTfLiteOk) {
//         std::cout
//             << "AllocateConcateTensors after HeightPartitioning returned ERROR"
//             << "\n";
//         return kTfLiteError;
//       }
//       if (!working_subgraph->IsInvokable()) {
//         if (working_subgraph->AllocateTensors() != kTfLiteOk)
//           return kTfLiteError;
//       }
//     }
//   }
//   return kTfLiteOk;
// }

TfLiteStatus Interpreter::AllocateTensorsofSubsets(int level, int model_id) {
  auto HeightPartitionandAllocateIfNeed = [&](Subgraph* subgraph) {
    std::cout << "HeightPartitionandAllocateIfNeed" << "\n";
    std::cout << "level "<< subgraph->GetLevel() <<  "id : " << subgraph->GetGraphid()
              << " resource : " << subgraph->GetResourceType() << "\n"; 
    if (subgraph->GetResourceType() == ResourceType::CO_CPU ||
        subgraph->GetResourceType() == ResourceType::CO_CPU_XNN ||
        subgraph->GetResourceType() == ResourceType::CO_GPU) {
      if (subgraph->GetPartitioningType() ==
          PartitioningType::HEIGHT_PARTITIONING) {
        std::cout << "Height partition on subgraph " << subgraph->GetGraphid()
                  << "\n";
        if (subgraph->PartitionHeightTest() != kTfLiteOk) {
          std::cout << "Height partitioning TEST returned ERROR"
                    << "\n";
          return kTfLiteError;
        }
        if (subgraph->AllocateTensors() != kTfLiteOk) {
          std::cout << "AllocateTensors after HeightPartitioning returned ERROR"
                    << "\n";
          return kTfLiteError;
        }
        if (subgraph->AllocateConcateTensors() != kTfLiteOk) {
          std::cout << "AllocateConcateTensors after HeightPartitioning "
                       "returned ERROR"
                    << "\n";
          return kTfLiteError;
        }
      }
    }
    return kTfLiteOk;
  };
  std::cout << "Allocate level " << level << "\n";
  std::cout << "Subset size " << subgraph_subsets.size() << "\n";
  Subgraph* primary_working_subgraph;
  if(subgraph_subsets.size() <= level){
        std::cout << "Interpreter : no registerd subgraph of "
                  << "model id [" << model_id << "] in level [" << level 
                  << "] no allocation occurs.\n";
        return kTfLiteOk;
  }
  for (auto subset : subgraph_subsets[level]) {
    if (subset.first == model_id) {
      if (subset.second.size() > 0) {
        std::cout << "Search subgraph id " << subset.second[0] << "\n";
        primary_working_subgraph = subgraph_id(subset.second[0]);  // Allocate first subgraph
        std::cout << "Allocate subgraph id " << subset.second[0] << "\n";
        if (primary_working_subgraph->AllocateTensors() != kTfLiteOk) {
          std::cout << "AllocateTensors of graph [" << subset.second[0] << "] "
                    << "returned ERROR"
                    << "\n";
          return kTfLiteError;
        }
        // Resize intermediate shared tensors with GetIntermediateTensorRange()
        int input_tensor_begin_idx, input_tensor_end_idx;
        if (GetIntermediateTensorRangeWithGraphSubset(
                model_id, &input_tensor_begin_idx, &input_tensor_end_idx) !=
            kTfLiteOk) {
          std::cout << "GetIntermediateTensorRangeWithGraphSubset ERROR"
                    << "\n";
          return kTfLiteError;
        }
        // // Debug print
        // for(int k=0; k<shared_tensor_and_graph.size(); ++k){
        //   if(shared_tensor_and_graph[k]->model_id == model_id){
        //     for(int i=0;
        //     i<shared_tensor_and_graph[k]->pair_tensor_graph.size(); ++i){
        //       std::cout << "tensor : " <<
        //       shared_tensor_and_graph[k]->pair_tensor_graph[i].first << "\n";
        //       std::cout << "subgraphs : ";
        //       for(int j=0;
        //       j<shared_tensor_and_graph[k]->pair_tensor_graph[i].second.size();
        //       ++j){
        //         std::cout <<
        //         shared_tensor_and_graph[k]->pair_tensor_graph[i].second[j] <<
        //         " ";
        //       }
        //       std::cout << "\n";
        //     }
        //   }
        // }
        
        // Description
        // First, propagate primiray subgraphs output dims to other subgraphs.
        // than allocate tensors.
        if(shared_tensor_and_graph.size() < level){
          std::cout << "AllocateTensorsofSubsets shared_tensor_and_graph ERROR" << "\n";
          return kTfLiteError;
        }
        for (auto shared_tensor_and_graph_ : shared_tensor_and_graph[level]) {
          if (shared_tensor_and_graph_->model_id == model_id) {
            for (int i = 0;
                 i < shared_tensor_and_graph_->pair_tensor_graph.size(); ++i) {
              int base_tensor =
                  shared_tensor_and_graph_->pair_tensor_graph[i].first;
              if (base_tensor >= input_tensor_begin_idx &&
                  base_tensor <= input_tensor_end_idx) {
                TfLiteTensor* working_tensor;
                std::vector<int> match_dims;
                for (int j = 0;
                     j < shared_tensor_and_graph_->pair_tensor_graph[i]
                             .second.size();
                     ++j) {
                  int working_subgraph =
                      shared_tensor_and_graph_->pair_tensor_graph[i].second[j];
                  if (j == 0) {
                    subgraph_id(working_subgraph)->PushToOutputs(base_tensor);
                    working_tensor =
                        subgraph_id(working_subgraph)->tensor(base_tensor);
                    match_dims = subgraph_id(working_subgraph)
                                     ->GetTensorShape(base_tensor);
                  } else {
                    // MUST FIX. REDUNDUNT INPUT, OUTPUT TENSOR!!
                    // MUST FIX. REDUNDUNT INPUT, OUTPUT TENSOR!!
                    subgraph_id(working_subgraph)
                        ->ResizeInputTensor(base_tensor, match_dims);
                    subgraph_id(working_subgraph)->PushToInputs(base_tensor);
                  }
                  if (subgraph_id(working_subgraph)->AllocateTensors() !=
                      kTfLiteOk)
                    return kTfLiteError;
                }
                working_tensor = nullptr;
                match_dims.clear();
              }
            }
          }
        }
      } else {
        std::cout << "Interpreter : no registerd subgraph of "
                  << "model id [" << model_id << "] no allocation occurs.\n";
        return kTfLiteOk;
      }
    }
    // Second, partition subgraphs and allocate again.
    for (int subgraph_idx = 0; subgraph_idx < subset.second.size();
         ++subgraph_idx) {
      int working_subgraph_id = subset.second[subgraph_idx];
      Subgraph* working_subgraph = subgraph_id(working_subgraph_id);
      if (HeightPartitionandAllocateIfNeed(working_subgraph) != kTfLiteOk) {
        std::cout << "HeightPartitionAndAllocateIfNeed returned ERROR"
                  << "\n";
        return kTfLiteError;
      }
      // Minsung
      if (working_subgraph->AllocateConcateTensors() != kTfLiteOk) {
        std::cout
            << "AllocateConcateTensors after HeightPartitioning returned ERROR"
            << "\n";
        return kTfLiteError;
      }
      if (!working_subgraph->IsInvokable()) {
        if (working_subgraph->AllocateTensors() != kTfLiteOk)
          return kTfLiteError;
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::GetIntermediateTensorRangeWithGraphSubset(
    int model_id, int* begin, int* end) {
  std::cout << "GetIntermediateTensorRangeWithGraphSubset" << "\n";
  TfLiteIntArray* execution_plan;
  int input_subgraph_id;
  int last_subgraph_id;
  primary_subgraph_->GetExecutionPlanSafe(&execution_plan);
  if (execution_plan->size < 0) {
    std::cout << "[ERROR] Execution Plan size < 0 in subgraph 0"
              << "\n";
    return kTfLiteError;
  }
  TfLiteNode* node;
  TfLiteRegistration* registration;
  // First, get first node's output tensor index.
  int first_execution_plan = execution_plan->data[0];
  if (primary_subgraph_->GetNodeAndRegistration(first_execution_plan, &node,
                                                &registration) != kTfLiteOk)
    return kTfLiteError;
  *begin = node->outputs->data[0];
  // Get last node's input tensor index from final subgraph.
  // Because the node's intput & output tensor(which are intermediate tensors)
  // WARNING!!!
  // MUST FIX FOR CASES NOT IN ASCENDING ORDER.
  // MUST FIX FOR CASES NOT IN ASCENDING ORDER.
  // idices are in ascending order, we can get intermediate tensors range.
  if (subgraphs_size() > 1) {  // If interpreter has more than one subgraph.
    TfLiteIntArray* execution_plan;
    primary_subgraph_->GetExecutionPlanSafe(&execution_plan);
    int last_execution_plan = execution_plan->data[execution_plan->size - 1];
    if (primary_subgraph_->GetNodeAndRegistration(last_execution_plan, &node,
                                                  &registration) != kTfLiteOk)
      return kTfLiteError;
    *end = node->outputs->data[0];
    TfLiteIntArrayFree(execution_plan);
  } else {  // Or interpreter has only one subgraph
    int last_execution_plan = execution_plan->data[execution_plan->size - 1];
    if (primary_subgraph_->GetNodeAndRegistration(last_execution_plan, &node,
                                                  &registration) != kTfLiteOk)
      return kTfLiteError;
    *end = node->outputs->data[0];
  }
  TfLiteIntArrayFree(execution_plan);
  std::cout << "GetIntermediateTensorRangeWithGraphSubset done" << "\n";
  return kTfLiteOk;
}

void Interpreter::ReserveNodes(int count) {
  primary_subgraph().ReserveNodes(count);
}

/// multi-lvel modified.
void Interpreter::AddSubgraphs(int subgraphs_to_add,
                               int* first_new_subgraph_index) {
  const size_t base_index = subgraphs__[0].size();
  if (first_new_subgraph_index) *first_new_subgraph_index = base_index;

  subgraphs__[0].reserve(base_index + subgraphs_to_add);
  for (int i = 0; i < subgraphs_to_add; ++i) {
    Subgraph* subgraph = new Subgraph(error_reporter_, external_contexts_,
                                      &subgraphs__[0], &resources_);
    subgraphs__[0].emplace_back(subgraph);
  }
}

TfLiteStatus Interpreter::AddNodeWithParameters(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const char* init_data, size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  return primary_subgraph().AddNodeWithParameters(
      inputs, outputs, {}, init_data, init_data_size, builtin_data,
      registration, node_index);
}

TfLiteStatus Interpreter::ResizeInputTensor(int tensor_index,
                                            const std::vector<int>& dims) {
  return primary_subgraph().ResizeInputTensor(tensor_index, dims);
}

TfLiteStatus Interpreter::ResizeInputTensorStrict(
    int tensor_index, const std::vector<int>& dims) {
  return primary_subgraph().ResizeInputTensorStrict(tensor_index, dims);
}

TfLiteStatus Interpreter::ReleaseNonPersistentMemory() {
  // TODO(b/138790287): We could do this for all subgraphs whose tensors have
  // been allocated. However, AllocateTensors() relies on Control Flow ops to
  // allocate tensors on 'children' subgraphs. Revisit this if required.
  return primary_subgraph().ReleaseNonPersistentMemory();
}

TfLiteStatus Interpreter::Invoke() {
  ScopedRuntimeInstrumentationProfile scoped_runtime_event(installed_profiler_,
                                                           "invoke");
  TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
      scoped_runtime_event, primary_subgraph().Invoke());

  if (!allow_buffer_handle_output_) {
    for (int tensor_index : outputs()) {
      TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
          scoped_runtime_event,
          primary_subgraph().EnsureTensorDataIsReadable(tensor_index));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Interpreter::AddTensors(int tensors_to_add,
                                     int* first_new_tensor_index) {
  return primary_subgraph().AddTensors(tensors_to_add, first_new_tensor_index);
}

TfLiteStatus Interpreter::ResetVariableTensors() {
  return primary_subgraph().ResetVariableTensors();
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    const char* buffer, size_t bytes, const Allocation* allocation) {
  return primary_subgraph().SetTensorParametersReadOnly(
      tensor_index, type, name, dims.size(), dims.data(), quantization, buffer,
      bytes, allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    bool is_variable) {
  return primary_subgraph().SetTensorParametersReadWrite(
      tensor_index, type, name, dims.size(), dims.data(), quantization,
      is_variable);
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, const char* buffer,
    size_t bytes, const Allocation* allocation) {
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return primary_subgraph().SetTensorParametersReadOnly(
      tensor_index, type, name, rank, dims, new_quantization, buffer, bytes,
      allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, bool is_variable,
    const size_t rank_dims_signature, const int* dims_signature) {
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return primary_subgraph().SetTensorParametersReadWrite(
      tensor_index, type, name, rank, dims, new_quantization, is_variable,
      rank_dims_signature, dims_signature);
}

TfLiteStatus Interpreter::SetExecutionPlan(const std::vector<int>& new_plan) {
  return primary_subgraph().SetExecutionPlan(new_plan);
}

void Interpreter::UseNNAPI(bool enable) {
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                       "Interpreter::UseNNAPI() is deprecated. Use "
                       "tflite::NnApiDelegate() directly instead.");
  primary_subgraph().UseNNAPI(enable);
}

// Multi-level modified
TfLiteStatus Interpreter::SetNumThreads(int num_threads) {
  if (num_threads < -1) {
    context_->ReportError(context_,
                          "num_threads should be >=0 or just -1 to let TFLite "
                          "runtime set the value.");
    return kTfLiteError;
  }
  for(auto& graphs : subgraphs__){
    for (auto& subgraph : graphs) {
      subgraph->context()->recommended_num_threads = num_threads;
    }
  }

  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    auto* c = external_contexts_[i];
    if (c && c->Refresh) {
      c->Refresh(context_);
    }
  }
  return kTfLiteOk;
}

// Multi-level modified
void Interpreter::SetAllowFp16PrecisionForFp32(bool allow) {
  for(auto& graphs : subgraphs__){
    for (auto& subgraph : graphs) {
      subgraph->context()->allow_fp32_relax_to_fp16 = allow;
    }
  }
}

// TODO(b/121264966): Subgraphs added after cancellation is set will not get the
// cancellation function added to their context.
// Multi-level modified
void Interpreter::SetCancellationFunction(void* data,
                                          bool (*check_cancelled_func)(void*)) {
  for (auto& graphs : subgraphs__){
    for (auto& subgraph : graphs) {
      subgraph->SetCancellationFunction(data, check_cancelled_func);
    }
  }                                         
}

bool Interpreter::IsCancelled() { return primary_subgraph().IsCancelled(); }

// Multi-level modified
TfLiteStatus Interpreter::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
  TfLiteStatus status = kTfLiteOk;
  for (auto& graphs : subgraphs__){
    for (auto& subgraph : graphs) {
      status = subgraph->ModifyGraphWithDelegate(delegate);
      if (status != kTfLiteOk) {
        break;
      }
    }
  }
  // Delegate-specific errors can be recovered from by restoring Interpreter to
  // its original state.
  if (status == kTfLiteDelegateError) {
    TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
  }
  return status;
}

TfLiteStatus Interpreter::ModifyGraphWithDelegateImpl(int graph_id) {
  TfLiteStatus status = kTfLiteOk;
  std::cout << "Delegate graph_id : " << graph_id << "\n";
  if (!delegates_prepared.empty()) {
    // delegate gpu
    if (subgraph_id(graph_id)->GetResourceType() == GPU ||
        subgraph_id(graph_id)->GetResourceType() == CO_GPU){
      status =
          subgraph_id(graph_id)->ModifyGraphWithDelegate(
            SearchAndReturnProperDelegate(DelegateType::GPU_DELEGATE, 0));
    }
    // delegate xnn
    else if (subgraph_id(graph_id)->GetResourceType() == CPU_XNN){
      int prefered_utilization = 0;
        prefered_utilization = subgraph_id(graph_id)->GetPartitioningRatio();
      std::cout << "prefered_utilization " << prefered_utilization << "\n";
      status = 
        subgraph_id(graph_id)->ModifyGraphWithDelegate(
          SearchAndReturnProperDelegate(DelegateType::XNN_DELEGATE, prefered_utilization));
    }else if (subgraph_id(graph_id)->GetResourceType() == CO_CPU_XNN){
      int prefered_utilization = 0;
        // prefered_utilization = subgraph_id(graph_id)->GetPartitioningRatio();
      std::cout << "prefered_utilization " << prefered_utilization << "\n";
      status = 
        subgraph_id(graph_id)->ModifyGraphWithDelegate(
          SearchAndReturnProperDelegate(DelegateType::XNN_DELEGATE, prefered_utilization));
    }
  } else {
    std::cout << "No delegate exists in this interpreter"
              << "\n";
    return kTfLiteError;
  }
  if (status != kTfLiteOk) return status;
  return status;
}
// Minsung
TfLiteStatus Interpreter::RegisterDelegate(DelegateWrapper* new_delegate) {
  std::cout << "register delegate" << "\n";
  delegates_prepared.push_back(new_delegate);
  return kTfLiteOk;
}

TfLiteDelegate* Interpreter::SearchAndReturnProperDelegate(DelegateType type, int utilization){
  for(int i=0; i<delegates_prepared.size(); ++i){
    if(delegates_prepared[i]->prefered_utilization == utilization
        && delegates_prepared[i]->delegate_type == type){
      std::cout << "delegate i " << i << " type : " << type << " util " << utilization << "\n";
      return delegates_prepared[i]->delegate;
    }
    
  }
  // for(auto delegation : delegates_prepared){
  //   if(delegation->prefered_utilization == utilization
  //       && delegation->delegate_type == type){
  //     return delegation->delegate;
  //   }
  // }
  return nullptr;
}

// Multi-level modified
TfLiteStatus Interpreter::RemoveAllDelegates() {
  for (auto& graphs : subgraphs__){
    for (auto& subgraph : graphs) {
      TF_LITE_ENSURE_STATUS(subgraph->RemoveAllDelegates());
    }
  }
  return kTfLiteOk;
}

bool Interpreter::HasDelegates() { return primary_subgraph().HasDelegates(); }

TfLiteStatus Interpreter::SetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteDelegate* delegate) {
  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  std::vector<TfLiteTensor>& tensors = primary_subgraph().tensors();
  TfLiteTensor* tensor = &tensors[tensor_index];

  TF_LITE_ENSURE(context_,
                 tensor->delegate == nullptr || tensor->delegate == delegate);
  tensor->delegate = delegate;
  if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
    TF_LITE_ENSURE(context_, tensor->delegate->FreeBufferHandle != nullptr);
    tensor->delegate->FreeBufferHandle(context_, tensor->delegate,
                                       &tensor->buffer_handle);
  }
  tensor->buffer_handle = buffer_handle;

  return kTfLiteOk;
}

TfLiteStatus Interpreter::GetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle* buffer_handle,
                                          TfLiteDelegate** delegate) {
  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  std::vector<TfLiteTensor>& tensors = primary_subgraph().tensors();
  TfLiteTensor* tensor = &tensors[tensor_index];

  *delegate = tensor->delegate;
  *buffer_handle = tensor->buffer_handle;

  return kTfLiteOk;
}

void Interpreter::SetProfiler(Profiler* profiler) {
  // Release resources occupied by owned_profiler_ which is replaced by
  // caller-owned profiler.
  owned_profiler_.reset(nullptr);
  installed_profiler_ = profiler;
  SetSubgraphProfiler();
}

void Interpreter::SetProfiler(std::unique_ptr<Profiler> profiler) {
  owned_profiler_ = std::move(profiler);
  installed_profiler_ = owned_profiler_.get();
  SetSubgraphProfiler();
}


// Multi-level modified
void Interpreter::SetSubgraphProfiler() {
  for (int subgraph_index = 0; subgraph_index < subgraphs_size();
       ++subgraph_index) {
    subgraph(subgraph_index)->SetProfiler(installed_profiler_,
                                            subgraph_index);
  }
}

Profiler* Interpreter::GetProfiler() {
  return primary_subgraph().GetProfiler();
}

//Multi-level modified [deprecated]
// tflite::Subgraph* Interpreter::CreateSubgraph() {
//   return new Subgraph(error_reporter_, external_contexts_, &subgraphs__[0],
//                       &resources_);
// }

tflite::Subgraph* Interpreter::CreateSubgraphInLevel(int level) {
  if(subgraphs__.size() <= level){
    subgraphs__.push_back(std::vector<std::unique_ptr<Subgraph>>());
  }
  return new Subgraph(error_reporter_, external_contexts_, &subgraphs__[level],
                      &resources_);
}


TfLiteTensor* Interpreter::input_tensor_of_model(int model_id) {
  for (auto subgraph_subset : subgraph_subsets[0]) {
    if (subgraph_subset.first == model_id) {
      Subgraph* graph = subgraph_id(subgraph_subset.second.at(0));
      if (graph == nullptr) return nullptr;
      int input_tensor_idx = graph->GetInputTensorIndex();
      return graph->tensor(input_tensor_idx);
    }
  }
}

TfLiteTensor* Interpreter::input_tensor_of_model(int level, int model_id) {
  if(subgraph_subsets.size() < level)
    return nullptr;
  for (auto subgraph_subset : subgraph_subsets[level]) {
    if (subgraph_subset.first == model_id) {
      Subgraph* graph = subgraph_id(subgraph_subset.second.at(0));
      if (graph == nullptr) return nullptr;
      int input_tensor_idx = graph->GetInputTensorIndex();
      return graph->tensor(input_tensor_idx);
    }
  }
}

void Interpreter::PrintSubgraphInfo() {
  std::cout << "============== Print Subgraph Info ===============" << "\n";
  std::cout << "Interpreter: subgraph size:" << subgraphs_size() << "\n";
  for(int level=0; level < subgraphs__.size(); ++level){
    std::cout << "Level : " << level << "\n";
    for(int index=0; index < subgraphs__[level].size(); ++index){
      std::cout << "ID : " << subgraphs__[level][index]->GetGraphid()
                << " model : " << subgraphs__[level][index]->GetModelid() << "\n";
    }
  }
  std::cout << "=================================================" << "\n";
  // for (int i = 0; i < subgraphs_size(); ++i) {
  //   std::cout << "id : " << subgraph(i)->GetGraphid()
  //             << " model : " << subgraph(i)->GetModelid() << "\n";
  // }
}

void Interpreter::SaveOriginTensorDims(Subgraph* origin_graph) {
  for (const auto tensor : origin_graph->tensors()) {
    TfLiteIntArray* new_ary = tensor.dims ? 
      TfLiteIntArrayCreate(tensor.dims->size) : TfLiteIntArrayCreate(0);
    for (int i = 0; tensor.dims && i < tensor.dims->size; ++i)
      new_ary->data[i] = tensor.dims->data[i];
    tensor_origin_dims.push_back(new_ary);
  }
}

TfLiteStatus Interpreter::AddNewSubgraph(int level, tflite::Subgraph* new_subgraph) {
  if(subgraphs__.size() <= level){
    std::cout << "Created new subgraph level " << level << "\n";
    subgraphs__.push_back(std::vector<std::unique_ptr<Subgraph>>());
  }
  if(new_subgraph != nullptr){
    subgraphs__[level].emplace_back(new_subgraph);
  }
  return kTfLiteOk;
}

// TfLiteStatus Interpreter::RegisterSubgraphSubsets(
//     tflite::Subgraph* new_subgraph) {
//   if (subgraph_subsets
//           .empty()) {  // if subgraph subset is empty, create new one
//     std::pair<int, std::vector<int>> new_subset;
//     new_subset.first = new_subgraph->GetModelid();
//     new_subset.second.push_back(new_subgraph->GetGraphid());
//     subgraph_subsets.push_back(new_subset);
//     return kTfLiteOk;
//   }
//   for (size_t j = 0; j < subgraph_subsets.size(); ++j) {
//     bool register_needed = false;
//     if (subgraph_subsets[j].first ==
//         new_subgraph->GetModelid()) {  // if a same model id exists.
//       for (size_t i = 0; i < subgraph_subsets[j].second.size(); ++i) {
//         if (subgraph_subsets[j].second[i] == new_subgraph->GetGraphid()) {
//           break;  // subgraph already registered.
//         }
//         if (i == subgraph_subsets[j].second.size() -
//                      1) {  // subgraph not registered.
//           register_needed = true;
//         }
//       }
//       if (register_needed) {
//         subgraph_subsets[j].second.push_back(new_subgraph->GetGraphid());
//         return kTfLiteOk;
//       }
//     }
//   }
//   // if there is no same model id in subsets, register new one.
//   std::pair<int, std::vector<int>> new_subset;
//   new_subset.first = new_subgraph->GetModelid();
//   new_subset.second.push_back(new_subgraph->GetGraphid());
//   subgraph_subsets.push_back(new_subset);
//   return kTfLiteOk;
// }

TfLiteStatus Interpreter::RegisterSubgraphSubsets(
    int level, tflite::Subgraph* new_subgraph) {
  if(subgraph_subsets.size() <= level){ // create new level.
    std::vector<std::pair<int, std::vector<int>>> new_level;
    subgraph_subsets.push_back(new_level);
    // register first subgraph of new level.
    std::pair<int, std::vector<int>> new_subset;
    new_subset.first = new_subgraph->GetModelid();
    new_subset.second.push_back(new_subgraph->GetGraphid());
    subgraph_subsets[level].push_back(new_subset);
    return kTfLiteOk;
  }
  // register next subgraphs.
  for (size_t j = 0; j < subgraph_subsets[level].size(); ++j) {
    bool register_needed = false;
    if (subgraph_subsets[level][j].first == new_subgraph->GetModelid()) {  // if a same model id exists.
      for (size_t i = 0; i < subgraph_subsets[level][j].second.size(); ++i) {
        if (subgraph_subsets[level][j].second[i] == new_subgraph->GetGraphid()) {
          break;  // subgraph already registered.
        }
        if (i == subgraph_subsets[level][j].second.size() - 1) {  // subgraph not registered.
          register_needed = true;
        }
      }
      if (register_needed) {
        subgraph_subsets[level][j].second.push_back(new_subgraph->GetGraphid());
        return kTfLiteOk;
      }
    }
  }
  return kTfLiteOk;
}


// Multi-level modified.
TfLiteStatus Interpreter::DeleteSubgraph(int subgraph_id) {
  for(size_t i = 0; i < subgraphs__.size(); ++i){
    for(size_t j = 0; j < subgraphs__[i].size(); ++j){
      if(subgraphs__[i][j]->GetGraphid() == subgraph_id){
        if(subgraph_id == 0){
            primary_subgraph_ = std::move(subgraphs__[i][j]);
        }
        subgraphs__[i].erase(subgraphs__[i].begin() + j);
      }
    }
  }
  
  for(auto& subsets : subgraph_subsets){
    for (size_t i = 0; i < subsets.size(); ++i) {
      for (size_t j = 0; j < subsets[i].second.size(); ++j) {
        if (subsets[i].second[j] == subgraph_id) {
          std::cout << "delete subgraph " << subgraph_id << " from subset" << "\n";
          subsets[i].second.erase(subsets[i].second.begin() + j);
          break;
        }
      }
    }
  }

  std::cout << "Subset has " << "\n";
  std::cout << "level size " << subgraph_subsets.size() << "\n";
  for(size_t level=0; level<subgraph_subsets.size(); ++level){
    std::cout << "subset size " << subgraph_subsets[level].size() << "\n";
    for(size_t model=0; model<subgraph_subsets[level].size(); ++model){
      std::cout << "subset graph size " << subgraph_subsets[level][model].second.size() << "\n";
      for(size_t graph=0; graph<subgraph_subsets[level][model].second.size(); ++graph){
        std::cout << "level " << level << " model " << subgraph_subsets[level][model].first
                  << " id " << subgraph_subsets[level][model].second[graph] << "\n";
      }
      std::cout << "\n";
    }
  }
  return kTfLiteOk;
}


tflite::Subgraph* Interpreter::returnProfiledOriginalSubgraph(int id) {
  for(auto subset : subgraph_subsets){
    for (int i = 0; i < subset.size(); ++i) {
      for (int j = 0; j < subset[i].second.size(); ++j) {
        int graph_id = subset[i].second[j];
        Subgraph* working_graph = subgraph_id(graph_id);
        if (working_graph == nullptr) {
          std::cout << "Cannot get pointer to subgraph " << graph_id << "\n";
          return nullptr;
        }
        if (working_graph->GetModelid() != id) continue;
        if (working_graph->IsOriginalSubgraph()) {
          working_graph->SetProfiled();
          return working_graph;
        }
      }
    }
  }
  return nullptr;  // no more profiled original subgraph
}

void Interpreter::GetTotalSubgraphID(std::vector<int>& graph_ids) {
  graph_ids.clear();
  for (int i = 0; i < subgraphs_size(); ++i) {
    graph_ids.push_back(subgraph(i)->GetGraphid());
  }
}

void Interpreter::GetTotalSubgraphIDInLevel(int level, std::vector<int>& graph_ids) {
  graph_ids.clear();
  if(subgraphs__.size() < level){
    std::cout << "GetTotalSubgraphIDInLevel ERROR" << "\n";
    exit(-1);
  }
  std::cout << "level " << level << " has " << subgraphs__[level].size() << " ids \n";
  for (int i = 0; i < subgraphs__[level].size(); ++i) {
    std::cout << "Get id " << subgraphs__[level][i]->GetGraphid() << "\n";
    graph_ids.push_back(subgraphs__[level][i]->GetGraphid());
  }
}

}  // namespace tflite

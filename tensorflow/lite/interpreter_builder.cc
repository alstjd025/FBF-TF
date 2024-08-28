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
#include "tensorflow/lite/interpreter_builder.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/profiling/platform_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/shared_library.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/version.h"

// aligned_alloc is available (via cstdlib/stdlib.h) with C++17/C11.
#if __cplusplus >= 201703L || __STDC_VERSION__ >= 201112L
#if !defined(__ANDROID__) || __ANDROID_API__ >= 28
// Neither Apple nor Windows provide aligned_alloc.
#if !defined(__APPLE__) && !defined(_WIN32)
#define TFLITE_USE_STD_ALIGNED_ALLOC
#endif
#endif
#endif

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

namespace tflite {

namespace {

// Ensure that ErrorReporter is non-null.
ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
  return e ? e : DefaultErrorReporter();
}

template <typename T>
TfLiteStatus Copy(const T* data_ptr, TfLiteIntArray** arr) {
  if (data_ptr->values() == nullptr) {
    return kTfLiteError;
  }

  int size = data_ptr->values()->size();
  *arr = TfLiteIntArrayCreate(size);
  for (int i = 0; i < size; i++) {
    (*arr)->data[i] = static_cast<int>(data_ptr->values()->Get(i));
  }
  return kTfLiteOk;
}

TfLiteStatus ParseSparseIndexVector(const DimensionMetadata* src,
                                    TfLiteDimensionMetadata* tgt) {
  if (src->array_segments() == nullptr || src->array_indices() == nullptr) {
    return kTfLiteError;
  }
  TfLiteStatus status = kTfLiteOk;
  switch (src->array_segments_type()) {
    case SparseIndexVector_Int32Vector:
      status = Copy(src->array_segments_as_Int32Vector(), &tgt->array_segments);
      break;
    case SparseIndexVector_Uint16Vector:
      status =
          Copy(src->array_segments_as_Uint16Vector(), &tgt->array_segments);
      break;
    case SparseIndexVector_Uint8Vector:
      status = Copy(src->array_segments_as_Uint8Vector(), &tgt->array_segments);
      break;
    default:
      status = kTfLiteError;
      break;
  }
  if (status != kTfLiteOk) return status;

  switch (src->array_indices_type()) {
    case SparseIndexVector_Int32Vector:
      return Copy(src->array_indices_as_Int32Vector(), &tgt->array_indices);
    case SparseIndexVector_Uint16Vector:
      return Copy(src->array_indices_as_Uint16Vector(), &tgt->array_indices);
    case SparseIndexVector_Uint8Vector:
      return Copy(src->array_indices_as_Uint8Vector(), &tgt->array_indices);
    default:
      break;
  }
  return kTfLiteError;
}

}  // namespace

const char* kEmptyTensorName = "";

// Using weak symbols to create a delegate allows automatic injection of the
// delegate simply by adding it as a dependency.
// For flex delegate, see also the strong override in
// lite/delegates/flex/delegate.cc.
TFLITE_ATTRIBUTE_WEAK Interpreter::TfLiteDelegatePtr AcquireFlexDelegate() {
  auto acquire_flex_delegate_func =
      reinterpret_cast<Interpreter::TfLiteDelegatePtr (*)()>(
          SharedLibrary::GetSymbol("TF_AcquireFlexDelegate"));
  if (acquire_flex_delegate_func) {
    return acquire_flex_delegate_func();
  }

#if !defined(TFLITE_IS_MOBILE_PLATFORM)
  // Load TF_AcquireFlexDelegate() from _pywrap_tensorflow_internal.so if it is
  // available.
  const char* filename_pywrap_tensorflow_internal =
#if defined(_WIN32)
      "_pywrap_tensorflow_internal.pyd";
#elif defined(__APPLE__)
      "python/_pywrap_tensorflow_internal.so";
#else
      "_pywrap_tensorflow_internal.so";
#endif
  void* lib_tf_internal =
      SharedLibrary::LoadLibrary(filename_pywrap_tensorflow_internal);
  if (lib_tf_internal) {
    acquire_flex_delegate_func =
        reinterpret_cast<Interpreter::TfLiteDelegatePtr (*)()>(
            SharedLibrary::GetLibrarySymbol(lib_tf_internal,
                                            "TF_AcquireFlexDelegate"));
    if (acquire_flex_delegate_func) {
      return acquire_flex_delegate_func();
    }
  }
#endif  // !defined(TFLITE_IS_MOBILE_PLATFORM)

  return Interpreter::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

InterpreterBuilder::InterpreterBuilder(const FlatBufferModel& model,
                                       const OpResolver& op_resolver)
    : model_(model.GetModel()),
      op_resolver_(op_resolver),
      error_reporter_(ValidateErrorReporter(model.error_reporter())),
      allocation_(model.allocation()) {}

InterpreterBuilder::InterpreterBuilder(const ::tflite::Model* model,
                                       const OpResolver& op_resolver,
                                       ErrorReporter* error_reporter)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(ValidateErrorReporter(error_reporter)) {}


InterpreterBuilder::InterpreterBuilder(const FlatBufferModel& model,
                                      const OpResolver& op_resolver,
                                      const char* model_name,
                                      int model_id)
    : model_(model.GetModel()),
      op_resolver_(op_resolver),
      error_reporter_(DefaultErrorReporter()),
      //allocation_(model->allocation()), check how allocation initialized.
      model_id_(model_id),
      model_name_(model_name) {}


InterpreterBuilder::InterpreterBuilder(const FlatBufferModel& model,
                                      const OpResolver& op_resolver,
                                      Interpreter* interpreter,
                                      const char* model_name,
                                      int model_id, bool is_sub_interpreter)
    : model_(model.GetModel()),
      op_resolver_(op_resolver),
      error_reporter_(DefaultErrorReporter()),
      allocation_(model.allocation()), 
      interpreter_(interpreter),
      model_id_(model_id),
      model_name_(model_name),
      is_sub_interpreter(is_sub_interpreter){
        dummy_profile_ = new ProfileData;
      }



InterpreterBuilder::~InterpreterBuilder() {}

void InterpreterBuilder::CopyRawPartitioningPlan(
                                    std::vector<int>& raw_plan){
  ProfileData* dummy_profile = new ProfileData;
  
  // Parse raw plan from scheduler
  bool start_node_flag = true;
  bool node_flag = true;
  int partitioning_ratio, resource_plan, subgraph_idx;
  std::vector<int> node_subset;
  subgraph_idx = 0;
  for(int idx=0; idx<TF_P_PLAN_LENGTH; ++idx){
    int current_value = raw_plan[idx];
    if(current_value == PART_PARM_SEP_ENDP) { break; } // met EOF
    else if(current_value == PART_PARM_SEP_OP){ 
      node_flag = false;
    }else if(current_value == PART_PARM_SEP_RESROURCE){
      partitioning_ratio = raw_plan[idx -1];
      resource_plan = raw_plan[idx - 2];
      // copy node subset
      for(const auto value : node_subset){
        dummy_profile->layer_subsets[subgraph_idx].push_back(value);
      }
      dummy_profile->partitioning_ratios[subgraph_idx].push_back(partitioning_ratio);
      dummy_profile->subset_resource.push_back(static_cast<ResourceType>(resource_plan));
      subgraph_idx++;
      node_subset.clear();
      start_node_flag = true;
      node_flag = true;
    }else if (current_value == PART_PARM_SEP_SUBG){
      continue;
    }else{ // parse node subset
      if(start_node_flag){
        start_node_flag = false;
        dummy_profile->layer_subsets.push_back(std::vector<int>());
        dummy_profile->partitioning_ratios.push_back(std::vector<int>());
      }
      if(node_flag) { node_subset.push_back(current_value); }
    }
  }
  dummy_profiles_.push_back(dummy_profile);
}

TfLiteStatus InterpreterBuilder::BuildLocalIndexToRegistrationMapping(
                                    const ::tflite::Model* model,
                                    const OpResolver& op_resolver){
  TfLiteStatus status = kTfLiteOk;
  // Reset state.
  flatbuffer_op_index_to_registration_.clear();
  unresolved_custom_ops_.clear();

  auto opcodes = model->operator_codes();
  if (!opcodes) {
    return status;
  }
  int num_custom_ops = 0;
  for (const OperatorCode* opcode : *opcodes) {
    if (GetBuiltinCode(opcode) == BuiltinOperator_CUSTOM) {
      num_custom_ops++;
    }
  }
  unresolved_custom_ops_.reserve(num_custom_ops);
  for (const OperatorCode* opcode : *opcodes) {
    const TfLiteRegistration* registration = nullptr;
    status = GetRegistrationFromOpCode(opcode, op_resolver, error_reporter_,
                                       &registration);
    if (status != kTfLiteOk) {
      if (GetBuiltinCode(opcode) != BuiltinOperator_CUSTOM) {
        return status;
      }
      // If it's an unresolved custom op, allow it for now. It might be resolved
      // by a delegate later.
      if (!opcode->custom_code()) {
        error_reporter_->Report(
            "Operator with CUSTOM builtin_code has no custom_code.\n");
        return status;
      }
      const auto* op_name = opcode->custom_code()->c_str();
      unresolved_custom_ops_.push_back(CreateUnresolvedCustomOp(op_name));
      registration = &unresolved_custom_ops_.back();
      has_flex_op_ |= IsFlexOp(op_name);
      status = kTfLiteOk;
    }
    flatbuffer_op_index_to_registration_.push_back(registration);
  }
  return status;
}


TfLiteStatus InterpreterBuilder::BuildLocalIndexToRegistrationMapping() {
  TfLiteStatus status = kTfLiteOk;
  // Reset state.
  flatbuffer_op_index_to_registration_.clear();
  unresolved_custom_ops_.clear();

  auto opcodes = model_->operator_codes();
  if (!opcodes) {
    return status;
  }
  int num_custom_ops = 0;
  for (const OperatorCode* opcode : *opcodes) {
    if (GetBuiltinCode(opcode) == BuiltinOperator_CUSTOM) {
      num_custom_ops++;
    }
  }
  unresolved_custom_ops_.reserve(num_custom_ops);
  for (const OperatorCode* opcode : *opcodes) {
    const TfLiteRegistration* registration = nullptr;
    status = GetRegistrationFromOpCode(opcode, op_resolver_, error_reporter_,
                                       &registration);
    if (status != kTfLiteOk) {
      if (GetBuiltinCode(opcode) != BuiltinOperator_CUSTOM) {
        return status;
      }
      // If it's an unresolved custom op, allow it for now. It might be resolved
      // by a delegate later.
      if (!opcode->custom_code()) {
        error_reporter_->Report(
            "Operator with CUSTOM builtin_code has no custom_code.\n");
        return status;
      }
      const auto* op_name = opcode->custom_code()->c_str();
      unresolved_custom_ops_.push_back(CreateUnresolvedCustomOp(op_name));
      registration = &unresolved_custom_ops_.back();
      has_flex_op_ |= IsFlexOp(op_name);
      status = kTfLiteOk;
    }
    flatbuffer_op_index_to_registration_.push_back(registration);
  }
  return status;
}

namespace {
template <class T>
std::vector<int> FlatBufferIntArrayToVector(T* flat_array) {
  // Initialize shape of tensors with null shape. Empty vectors are converted
  // to nullptr for models that are constructed via flatbuffers::Pack.
  if (flat_array == nullptr) {
    return {};
  }
  std::vector<int> ret(flat_array->size());
  for (int i = 0; i < flat_array->size(); i++) {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

// Used to determine how the op data parsing function creates its working space.
class MallocDataAllocator : public BuiltinDataAllocator {
 public:
  void* Allocate(size_t size, size_t alignment_hint) override {
#ifdef TFLITE_USE_STD_ALIGNED_ALLOC
    // Ensure that alignment is a power of two and a multiple of sizeof(void *)
    // and that size is an integral multiple of alignment.
    size_t used_alignment = std::max(alignment_hint, sizeof(void*));
    size_t used_size =
        ((size + used_alignment - 1) / used_alignment) * used_alignment;
    TFLITE_DCHECK(
        (used_alignment != 0) &&
        ((used_alignment & (used_alignment - 1)) == 0));  // is power-of-two
    return aligned_alloc(used_alignment, used_size);
#else
    return malloc(size);
#endif
  }
  void Deallocate(void* data) override { free(data); }
};

}  // namespace

TfLiteStatus InterpreterBuilder::CreateSubgraphFromFlatBuffer(){
  // Creates an acient subgraph from a raw flatbuffer model.
  // This function follows the instructions below.
  // 1. Get a raw subgraph from flatbuffer model.
  // 2. Parse operators and tensors from it.
  // 3. Parse nodes and tensors to new subgraph from operators and tensors.
  // 4. Allocate tensors in new subgraph.
  // 5. Make subgraph with a new Job(default) struct(deprecated, subject to change).
  // 6. Store the Job and subgraph to given interpreter.

  if(!interpreter_){
    std::cout << "No interpreter ERROR" << "\n";
    return kTfLiteError;
  }

  if(!model_){
    std::cout << "No model ERROR" << "\n";
    return kTfLiteError;
  }

  if(BuildLocalIndexToRegistrationMapping() != kTfLiteOk){
    std::cout << "Registration Failed" << "\n";
    return kTfLiteError;
  }

  // Flatbuffer model schemas define a list of opcodes independent of the graph.
  // We first map those to registrations. This reduces string lookups for custom
  // ops since we only do it once per custom op rather than once per custom op
  // invocation in the model graph.
  // Construct interpreter with correct number of tensors and operators.
  auto* subgraphs = model_->subgraphs();
  auto* buffers = model_->buffers();
  int subgraph_index = 0;
  // Minsung
  // We assume that the raw model has only one subgraph.
  if (subgraphs->size() != 1) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Raw subgraph in the model Error.\n");
    return kTfLiteError;
  }

  if (!buffers) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No buffers in the model.\n");
    return kTfLiteError;
  }
  const tflite::SubGraph* subgraph = (*subgraphs)[subgraph_index];
  tflite::Subgraph* modified_subgraph = interpreter_->CreateSubgraphInLevel(0);

  auto operators = subgraph->operators();
  auto tensors = subgraph->tensors();
  if (!operators || !tensors) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                          "Did not get operators or tensors in subgraph %d.\n",
                          subgraph_index);
    return kTfLiteError;
  }
  if (modified_subgraph->AddTensors(tensors->size()) != kTfLiteOk) {
    return kTfLiteError;
  }
  
  // Set this subgraph as original one.
  // Original subgraph should be only one for a model at runtime.
  modified_subgraph->SetOriginalSubgraph();
  // Parse inputs/outputs
  modified_subgraph->SetInputs(
      FlatBufferIntArrayToVector(subgraph->inputs()));
  modified_subgraph->SetOutputs(
      FlatBufferIntArrayToVector(subgraph->outputs()));

  // Finally setup nodes and tensors
  if (ParseNodes(operators, modified_subgraph) != kTfLiteOk)
    return kTfLiteError;
  if (ParseTensors(buffers, tensors, modified_subgraph) != kTfLiteOk)
    return kTfLiteError;

  std::vector<int> variables;
  for (int i = 0; i < modified_subgraph->tensors_size(); ++i) {
    auto* tensor = modified_subgraph->tensor(i);
    if (tensor->is_variable) {
      variables.push_back(i);
    }
  }
  modified_subgraph->SetVariables(std::move(variables));
  std::cout << "Interpreterbuilder : created modified subgraph" << "\n";
  // Minsung
  // Is this necessary?
  if (num_fp32_tensors_ > 0) {
    (*interpreter_).lazy_delegate_providers_ =
        op_resolver_.GetDelegates(default_thread);
  }

  // Minsung
  // Is this necessary?
  if (ApplyDelegates(interpreter_, default_thread) != kTfLiteOk)
    return kTfLiteError;

  // Minsung
  // Allocate the new subgraph
  if(modified_subgraph->AllocateTensors() != kTfLiteOk){
    std::cout << "Subgraph allocation failed" << "\n";
    return kTfLiteError;
  }
  std::cout << "Interpreterbuilder : Allocated tensors" << "\n";
  std::vector<tflite::Subgraph*> subgraph_created;
  subgraph_created.push_back(modified_subgraph);
  // Minsung
  // Create a new job from subgraph.
  if(RegisterSubgraphToInterpreter(0, subgraph_created) != kTfLiteOk){
    std::cout << "RegisterSubgraphToInterpreter ERROR" << "\n";
    return kTfLiteError; 
  }
  std::cout << "Interpreterbuilder : Created subgraph with default job" << "\n";

  interpreter_->SaveOriginTensorDims (modified_subgraph);
  std::cout << "Interpreterbuilder : Registered default job and subgraph" << "\n";
  std::cout << "New Graph id : " << modified_subgraph->GetGraphid() << "\n";
  std::cout << "New Graph model id : " << modified_subgraph->GetModelid() << "\n";
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::CreateSubgraphForStress(){
  
}

// This function is deprecated since multi-level subgraph.
// This function is deprecated since multi-level subgraph.
// This function is deprecated since multi-level subgraph.
TfLiteStatus InterpreterBuilder::CreateSubgraphsFromParameter(
                                      tflite::Subgraph* profiled_subgraph){
  if(!profiled_subgraph->IsProfiled()){
    std::cout << "InterpreterBuilder : Subgraph is not profiled \n";
    return kTfLiteError;
  }
  
  // Now create subgraphs from original one.
  // 1. read profiled data
  // 2. make new subgraphs
  // 3. parse nodes and tensors from original nodes_and_registrations
  // 4. 

  auto* subgraphs = model_->subgraphs();
  auto* buffers = model_->buffers();
  if (subgraphs->size() == 0) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No subgraph in the model.\n");
    return kTfLiteError;
  }
  if (!buffers) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No buffers in the model.\n");
    return kTfLiteError;
  }
  int new_subgraph_index = 0;
  int count_node_per_subgraph = 0;
  int total_count_node_per_subgraph = 0; 
  int count_tensor_per_subgraph = 0;
  int total_count_tensor_per_subgraph = 0;
  std::vector<TfLiteIntArray*> inputs;
  std::vector<TfLiteIntArray*> outputs;
  std::vector<tflite::Subgraph*> subgraphs_created;
  for (int subgraph_index = 0; subgraph_index < subgraphs->size();
      ++subgraph_index) {
        // Note : Assume that we have only one subgraph before.
    const tflite::SubGraph* subgraph = (*subgraphs)[subgraph_index];
    auto operators = subgraph->operators();
    auto tensors = subgraph->tensors();
    // Look for Conv2d OPs and save tensor index
    std::vector<int> conv_idx;
    bool set_input = true;
    std::vector<int>* input_tensor = new std::vector<int>;
    std::vector<int>* output_tensor = new std::vector<int>;
    std::vector<int>* tensors_ = new std::vector<int>;
    //  std::vector<int>* tmp_input_tensor = new std::vector<int>;
    // std::vector<int>* tmp_output_tensor = new std::vector<int>;
    // std::vector<int>* tmp_tensors_ = new std::vector<int>;
    std::vector<SubgraphPartitioningPlan*> master_partitioning_plan;
    
    auto CreatePartitioningPlanFromProfile = 
        [&](const std::vector<ProfileData*>& profile){
      for(int k=0; k<profile.size(); ++k){
        for(int i=0; i<profile[k]->layer_subsets.size(); ++i){ //graphs
          SubgraphPartitioningPlan* new_plan = new SubgraphPartitioningPlan;
          new_plan->size = profile[k]->layer_subsets[i].size();
          new_plan->nodes = new int[new_plan->size];

          // Consider better implementation. (memory waste)
          new_plan->partitioning_ratios = new int[new_plan->size];

          // Set the resource type of subgraph.        
          // check if subset resource is Co_execution
          // if so, check this interpreterbuilder if it is co-execution builder.
          // co-execution builder : build cpu subgraphs for co-execution.
          // not co-execution builder : build gpu subgraphs for co-execution.
          // !! be carefull of redundant subgraphs.
          switch (profile[k]->subset_resource[i])
          {
          case TF_P_PLAN_CPU:
            if(is_sub_interpreter){
              // Don't create subgraph if sub-interprter. 
              new_plan->resource_type = ResourceType::NONE;
              break;
            }
            new_plan->resource_type = ResourceType::CPU;
            break;
          case TF_P_PLAN_GPU:
            if(is_sub_interpreter){
              // Don't create subgraph if sub-interprter. 
              new_plan->resource_type = ResourceType::NONE;
              break;
            }
            new_plan->resource_type = ResourceType::GPU;
            break;
          case TF_P_PLAN_CO_E:
            if(is_sub_interpreter){
              new_plan->resource_type = ResourceType::CO_CPU;
              break;
            }
            else
              new_plan->resource_type = ResourceType::CO_GPU;
            break;
          case TF_P_PLAN_CPU_XNN:
            if(is_sub_interpreter){
              // Don't create subgraph if sub-interprter. 
              new_plan->resource_type = ResourceType::NONE;
              break;
            }
            else
              new_plan->resource_type = ResourceType::CPU_XNN;
            break;
          case TF_P_PLAN_CO_E_XNN:
            if(is_sub_interpreter){
              new_plan->resource_type = ResourceType::CO_CPU_XNN;
              break;
            }
            else
              new_plan->resource_type = ResourceType::CO_GPU;
            break;

          default:
            break;
          }
          std::cout << "nodes in subgraph : ";
          for(int j=0; j<profile[k]->layer_subsets[i].size(); ++j){ //layers 
            new_plan->nodes[j] = profile[k]->layer_subsets[i][j]; 
            std::cout << new_plan->nodes[j] << " ";
            new_plan->partitioning_ratios[j] = profile[k]->partitioning_ratios[i][0];
          }
          // std::cout << "\n";
          // std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
          // std::cout << "HH\n";
          master_partitioning_plan.push_back(new_plan);
        }
        // std::cout << "ZZZ\n";
      }
      return;
    };
    CreatePartitioningPlanFromProfile(dummy_profiles_);

    // Initialize variables for graph & tensor profiling.
    std::vector<std::pair<int, std::vector<int>>> subgraph_and_tensors;
    std::vector<std::pair<int, std::vector<int>>> shared_info;
    int shared_tensor_bucket[tensors->size()][master_partitioning_plan.size()];
    for(int i=0; i<tensors->size(); ++i)
      for(int j=0; j<master_partitioning_plan.size(); ++j)
        shared_tensor_bucket[i][j] = 0;

    std::queue<tflite::Subgraph*> prev_queue;


    // EZE

    // Partitioning iteration begins
    for(int partition_itr=0; partition_itr<master_partitioning_plan.size(); 
                                                ++partition_itr){
      if(master_partitioning_plan[partition_itr]->resource_type 
          == ResourceType::NONE)
        continue; // Don't make subgraph for resourcetype::NONE
      /// Make a new subgraph
      tflite::Subgraph* new_subgraph = interpreter_->CreateSubgraphInLevel(0);
      subgraphs_created.push_back(new_subgraph);
      // std::cout << "LLLLLLLll " << subgraphs_created.size();
      if(!prev_queue.empty()){ // make linked-list structure of subgraphs
        prev_queue.front()->SetNextSubgraph(new_subgraph);
        new_subgraph->SetPrevSubgraph(prev_queue.front());
        prev_queue.pop();
      }
      prev_queue.push(new_subgraph);
      const int* nodes_in_partition = master_partitioning_plan[partition_itr]->nodes;
      const int num_nodes_in_partition = master_partitioning_plan[partition_itr]->size;
      
      switch (master_partitioning_plan[partition_itr]->resource_type)
      {
      case ResourceType::CPU:
        // Set this sugraph for cpu subgraph
        new_subgraph->SetResourceType(ResourceType::CPU);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);
        new_subgraph->context()->recommended_num_threads = 6;
        break;
      case ResourceType::GPU:
        // Set this sugraph for gpu subgraph
        new_subgraph->SetResourceType(ResourceType::GPU);
        new_subgraph->context()->recommended_num_threads = 6;  
        break;
      case ResourceType::CO_CPU:
        new_subgraph->SetResourceType(ResourceType::CO_CPU);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);  
        if(master_partitioning_plan[partition_itr]->partitioning_ratios[0] > 10)
          new_subgraph->SetPartitioningType(PartitioningType::HEIGHT_PARTITIONING);
        else
          new_subgraph->SetPartitioningType(PartitioningType::CHANNEL_PARTITIONING);
        new_subgraph->context()->recommended_num_threads = 6;
        break;
      case ResourceType::CO_GPU:
        new_subgraph->SetResourceType(ResourceType::CO_GPU);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);
        if(master_partitioning_plan[partition_itr]->partitioning_ratios[0] > 10)
          new_subgraph->SetPartitioningType(PartitioningType::HEIGHT_PARTITIONING);
        else
          new_subgraph->SetPartitioningType(PartitioningType::CHANNEL_PARTITIONING);        
          new_subgraph->context()->recommended_num_threads = 6;  
        break;
      case ResourceType::CPU_XNN:
        new_subgraph->SetResourceType(ResourceType::CPU_XNN);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);
        new_subgraph->context()->recommended_num_threads = 6;
        break;
      case ResourceType::CO_CPU_XNN:
        new_subgraph->SetResourceType(ResourceType::CO_CPU_XNN);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);
        if(master_partitioning_plan[partition_itr]->partitioning_ratios[0] > 10)
          new_subgraph->SetPartitioningType(PartitioningType::HEIGHT_PARTITIONING);
        else
          new_subgraph->SetPartitioningType(PartitioningType::CHANNEL_PARTITIONING);        
        break;
      default:
        break;
      }
      std::vector<int> nodes_to_parse;
      // Now setup nodes and tensors for new subgraph
      std::cout << std::endl;
      std::cout << "partitioning_iteration : " <<partition_itr << std::endl;  // EZE

      for(int j=0; j < num_nodes_in_partition; ++j){
        // std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
        // std::cout << "HHH : " << j << std::endl;  // 
        int working_op = nodes_in_partition[j];
        nodes_to_parse.push_back(working_op);
        const auto* op = operators->Get(working_op);
        int op_index = op->opcode_index();
        /// get every tensor indices of all nodes
        for(int k=0; k<FlatBufferIntArrayToVector(op->inputs()).size(); ++k)
          tensors_->push_back(FlatBufferIntArrayToVector(op->inputs())[k]);
        for(int k=0; k<FlatBufferIntArrayToVector(op->outputs()).size(); ++k)
          tensors_->push_back(FlatBufferIntArrayToVector(op->outputs())[k]);          
        /// input tensor should be first node's input tensor in partitioning plan
        if(j == 0){
          input_tensor = new std::vector<int>;
          input_tensor->push_back(FlatBufferIntArrayToVector(op->inputs())[0]);
          new_subgraph->SetActualInput(*input_tensor); // set 'actual' input tensors
        }  /// output tensor should be last node's output tensor in partitioning plan
        if(j == num_nodes_in_partition - 1){ // last iteration
          output_tensor = new std::vector<int>;
          output_tensor->push_back(FlatBufferIntArrayToVector(op->outputs())[0]);
          new_subgraph->SetActualOutput(*output_tensor); // set 'actual' output tensors
          if (new_subgraph->AddTensors(tensors->size()) != kTfLiteOk){
            return kTfLiteError;
          }
          std::pair<int, std::vector<int>> graph_and_tensors(partition_itr, *tensors_);
          subgraph_and_tensors.push_back(graph_and_tensors);
          new_subgraph->SetInputs(  // set 'all' input tensors
                      std::vector<int>(input_tensor->begin(), input_tensor->end()));
          new_subgraph->SetOutputs(  // set 'all' output tensors
                      std::vector<int>(output_tensor->begin(), output_tensor->end()));
          // if (ParseNodes(operators, new_subgraph,
          //                 nodes_in_partition[0], nodes_in_partition[j]) != kTfLiteOk)
          //   return kTfLiteError;
          if (ParseNodes(operators, new_subgraph, nodes_to_parse) != kTfLiteOk)
            return kTfLiteError;
          if (ParseTensors(buffers, tensors, new_subgraph, *tensors_) != kTfLiteOk)
            return kTfLiteError;
          std::vector<int> variables;
          for (int l = 0; l < new_subgraph->tensors_size(); ++l) {
            auto* tensor = new_subgraph->tensor(l);
            if (tensor->is_variable) {
              variables.push_back(l);
            }
          }
          new_subgraph->SetVariables(std::move(variables));
          std::cout << "Interpreterbuilder : created new subgraph" << "\n";
        }
      }
      // tmp_input_tensor = input_tensor;
      // tmp_output_tensor = output_tensor;
      // tmp_tensors_ = tensors_; 
      // EZE : Do not use this method
      // (1) Dynamic allocate each subgraph's tensor
      // (2) Check each subgraph's tensor. 
      // (3) if there are two subgraphs which use same tensor, Make free later subgraph's tensor
      //     and use former subgraphs' tensor.
      input_tensor->clear();
      delete input_tensor;
      tensors_->clear();
      delete tensors_;
      output_tensor->clear();
      delete output_tensor;
      tensors_ = new std::vector<int>;
      output_tensor = new std::vector<int>;
    }
    
    // tflite::Job* new_job = new tflite::Job;
    // if(BindSubgraphWithJob(subgraphs_created, new_job) !=
    //     kTfLiteOk){
    //   std::cout << "BindSubgraphWithJob ERROR" << "\n";
    //   return kTfLiteError;
    // }
    // if(RegisterJobAndSubgraphs(subgraphs_created, new_job) !=
    //     kTfLiteOk){
    //   std::cout << "RegisterJobAndSubgraphs ERROR" << "\n";
    //   return kTfLiteError;
    // }
    interpreter_->PrintSubgraphInfo(); 
    std::cout << "RegisterJobAndSubgraphs" << "\n";
    // Fill shared tensor bucket  
    for(size_t graph_idx=0; graph_idx<subgraph_and_tensors.size(); ++graph_idx){
      for(size_t j=0; j<subgraph_and_tensors[graph_idx].second.size(); ++j){
        int tensor = subgraph_and_tensors[graph_idx].second[j];
        if(shared_tensor_bucket[tensor][graph_idx] != 1)
          shared_tensor_bucket[tensor][graph_idx] = 1; // Tensor shared flag
      }
    }
    // Save shared intermediate tensor indices in interpreter's graph_and_shared_tensor.
    // Used when AllocateTensorsofAllSubgraphs called.
    // (to propagate tensor shapes)
    for(size_t t=0; t<tensors->size(); ++t){
      std::pair<int, std::vector<int>> pair_tensor_graph;
      std::vector<int> sharing_subgraph_id;
      for(size_t g=0; g<subgraph_and_tensors.size(); ++g){
        if(shared_tensor_bucket[t][g]){
          sharing_subgraph_id.push_back(subgraphs_created[g]->GetGraphid());
        }
      }
      if(sharing_subgraph_id.size() > 1){
        pair_tensor_graph.first = t;        // tensor index
        pair_tensor_graph.second = sharing_subgraph_id; // subgraph id
        shared_info.push_back(pair_tensor_graph);
      }
      sharing_subgraph_id.clear();
    }
    SharedTensorsInGraphs* temp = new SharedTensorsInGraphs;
    temp->pair_tensor_graph = shared_info;
    temp->model_id = model_id_;
    if((interpreter_)->shared_tensor_and_graph.empty()){
      (interpreter_)->shared_tensor_and_graph.push_back(std::vector<SharedTensorsInGraphs*>());
    }
    (interpreter_)->shared_tensor_and_graph[0].push_back(temp);
  }
  
  std::cout << "USE INTERPRETER API (before Deletesubgraph)\n";
  std::vector<int> subgraph_set;
  interpreter_->GetTotalSubgraphID(subgraph_set);
  std::cout << "subgraph_set size : " << subgraph_set.size() << std::endl;
  
  // Delete old subgraph and move it to interpreter's primary subgraph.
  if(interpreter_->DeleteSubgraph(profiled_subgraph->GetGraphid()) 
      != kTfLiteOk){
    std::cout << "DeleteSubgraph ERROR" << "\n";
    return kTfLiteError;
  }

 std::cout << "USE INTERPRETER API (After Deletesubgraph)\n";
  // std::vector<int> subgraph_set;
  interpreter_->GetTotalSubgraphID(subgraph_set);
  std::cout << "subgraph_set size : " << subgraph_set.size() << std::endl;

  // MUST CHECK
  // Does CPU-side interpreter need to call AllocateTensors twice?
  std::cout << "model_id : " << model_id_ << std::endl;
  if(interpreter_->AllocateTensorsofSubsets(0, model_id_) != kTfLiteOk){
    std::cout << "AllocateTensorsofSubsets ERROR" << "\n";
    return kTfLiteError;
  }
  std::cout << "Allocated tensors" << "\n";
  if(is_sub_interpreter){ // If Co-execution CPU InterpreterBuilder
    if(PartitionChannels((subgraphs_created)) != kTfLiteOk){
      std::cout << "Partition channel-wise for cpu ERROR" << "\n";
      return kTfLiteError;
    }
  }
  // printf("\033[0;31m<<<<<<<<<<SUBGRAPHS_CREATED_SIZE : %d>>>>>>>>>>\033[0m\n",subgraphs_created.size());
  // int tensor_output_id = -1;
  // ResourceType resource_type = ResourceType::CPU;
  // for (int k=0;k<subgraphs_created.size();k++){
  //   tensor_output_id = subgraphs_created[k]->GetFirstOutputTensorIndex();
  //   resource_type = subgraphs_created[k]->GetResourceType();
  //   for (int j=k-1;j>=0;j--){
  //     if ((tensor_output_id == subgraphs_created[j]->GetFirstOutputTensorIndex()) 
  //     && (resource_type == subgraphs_created[j]->GetResourceType())){
  //       std::cout << "FIND CANDIDATE subgraph : " << k;
  //       std::cout << ", AND CANDIDATE's mother subgraph : " << j;
  //       std::cout << " OVERALL tensor size : "<<subgraphs_created[j]->tensors_size() << std::endl;
  //       std::cout << " OUTPUT tensor id : "<< subgraphs_created[k]->GetFirstOutputTensorIndex()<< std::endl;
  //       std::cout << "Arena before" << subgraphs_created[k]->GetArenaRWBufferSize() << "\n";
  //       subgraphs_created[k]->FreeArenaAllocation();
  //       std::cout << "Arena after" << subgraphs_created[k]->GetArenaRWBufferSize() << "\n";
  //       for (int n=0;n<subgraphs_created[k]->tensors().size(); n++){          
  //           subgraphs_created[k]->tensors()[n].data.data = subgraphs_created[j]->tensors()[n].data.data;
  //           subgraphs_created[k]->tensors()[n].bytes = subgraphs_created[j]->tensors()[n].bytes;
  //           subgraphs_created[k]->tensors()[n].allocation_type = kTfLiteCustom;
  //       }
  //     }
  //   }
  // }
  // printf("\033[0;31m<<<<<<<<<<SUBGRAPHS_CREATED_SIZE : %d>>>>>>>>>>\033[0m\n",subgraphs_created.size());
  // /////////////////////////////////////////////////////////////////////////////////

  if(DelegateSubgraphs(subgraphs_created) != kTfLiteOk){
    std::cout << "DelegateOldSubgraphs ERROR" << "\n";
    return kTfLiteError;
  }
  std::cout << "Delegate tensors" << "\n";
  std::cout << "Interpreterbuilder: Subgraphs created" << "\n";
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::CreateSubgraphsFromParameter(
                                      int level,
                                      tflite::Subgraph* profiled_subgraph){
  if(level == 0 && !profiled_subgraph->IsProfiled()){
    std::cout << "InterpreterBuilder : Subgraph is not profiled \n";
    return kTfLiteError;
  }

  auto* subgraphs = model_->subgraphs();
  auto* buffers = model_->buffers();
  if (subgraphs->size() == 0) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No subgraph in the model.\n");
    return kTfLiteError;
  }
  if (!buffers) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No buffers in the model.\n");
    return kTfLiteError;
  }
  int new_subgraph_index = 0;
  int count_node_per_subgraph = 0;
  int total_count_node_per_subgraph = 0; 
  int count_tensor_per_subgraph = 0;
  int total_count_tensor_per_subgraph = 0;
  std::vector<TfLiteIntArray*> inputs;
  std::vector<TfLiteIntArray*> outputs;
  std::vector<tflite::Subgraph*> subgraphs_created;
  for (int subgraph_index = 0; subgraph_index < subgraphs->size();
      ++subgraph_index) {
        // Note : Assume that we have only one subgraph before.
    const tflite::SubGraph* subgraph = (*subgraphs)[subgraph_index];
    auto operators = subgraph->operators();
    auto tensors = subgraph->tensors();
    // Look for Conv2d OPs and save tensor index
    std::vector<int> conv_idx;
    bool set_input = true;
    std::vector<int>* input_tensor = new std::vector<int>;
    std::vector<int>* output_tensor = new std::vector<int>;
    std::vector<int>* tensors_ = new std::vector<int>;
    //  std::vector<int>* tmp_input_tensor = new std::vector<int>;
    // std::vector<int>* tmp_output_tensor = new std::vector<int>;
    // std::vector<int>* tmp_tensors_ = new std::vector<int>;
    std::vector<SubgraphPartitioningPlan*> master_partitioning_plan;
    
    auto CreatePartitioningPlanFromProfile = 
        [&](const std::vector<ProfileData*>& profile){
      for(int k=0; k<profile.size(); ++k){
        for(int i=0; i<profile[k]->layer_subsets.size(); ++i){ //graphs
          SubgraphPartitioningPlan* new_plan = new SubgraphPartitioningPlan;
          new_plan->size = profile[k]->layer_subsets[i].size();
          new_plan->nodes = new int[new_plan->size];

          // Consider better implementation. (memory waste)
          new_plan->partitioning_ratios = new int[new_plan->size];

          // Set the resource type of subgraph.        
          // check if subset resource is Co_execution
          // if so, check this interpreterbuilder if it is co-execution builder.
          // co-execution builder : build cpu subgraphs for co-execution.
          // not co-execution builder : build gpu subgraphs for co-execution.
          // !! be carefull of redundant subgraphs.
          switch (profile[k]->subset_resource[i])
          {
          case TF_P_PLAN_CPU:
            if(is_sub_interpreter){
              // Don't create subgraph if sub-interprter. 
              new_plan->resource_type = ResourceType::NONE;
              break;
            }
            new_plan->resource_type = ResourceType::CPU;
            break;
          case TF_P_PLAN_GPU:
            if(is_sub_interpreter){
              // Don't create subgraph if sub-interprter. 
              new_plan->resource_type = ResourceType::NONE;
              break;
            }
            new_plan->resource_type = ResourceType::GPU;
            break;
          case TF_P_PLAN_CO_E:
            if(is_sub_interpreter){
              new_plan->resource_type = ResourceType::CO_CPU;
              break;
            }
            else
              new_plan->resource_type = ResourceType::CO_GPU;
            break;
          case TF_P_PLAN_CPU_XNN:
            if(is_sub_interpreter){
              // Don't create subgraph if sub-interprter. 
              new_plan->resource_type = ResourceType::NONE;
              break;
            }
            else
              new_plan->resource_type = ResourceType::CPU_XNN;
            break;
          case TF_P_PLAN_CO_E_XNN:
            if(is_sub_interpreter){
              new_plan->resource_type = ResourceType::CO_CPU_XNN;
              break;
            }
            else
              new_plan->resource_type = ResourceType::CO_GPU;
            break;

          default:
            break;
          }
          std::cout << "nodes in subgraph : ";
          for(int j=0; j<profile[k]->layer_subsets[i].size(); ++j){ //layers 
            new_plan->nodes[j] = profile[k]->layer_subsets[i][j]; 
            std::cout << new_plan->nodes[j] << " ";
            new_plan->partitioning_ratios[j] = profile[k]->partitioning_ratios[i][0];
          }
          master_partitioning_plan.push_back(new_plan);
        }
      }
      return;
    };
    CreatePartitioningPlanFromProfile(dummy_profiles_);

    // Initialize variables for graph & tensor profiling.
    std::vector<std::pair<int, std::vector<int>>> subgraph_and_tensors;
    std::vector<std::pair<int, std::vector<int>>> shared_info;
    int shared_tensor_bucket[tensors->size()][master_partitioning_plan.size()];
    for(int i=0; i<tensors->size(); ++i)
      for(int j=0; j<master_partitioning_plan.size(); ++j)
        shared_tensor_bucket[i][j] = 0;

    std::queue<tflite::Subgraph*> prev_queue;

    // Partitioning iteration begins
    for(int partition_itr=0; partition_itr<master_partitioning_plan.size(); 
                                                ++partition_itr){
      if(master_partitioning_plan[partition_itr]->resource_type == ResourceType::NONE){
        continue; // Don't make subgraph for resourcetype::NONE
      }
      /// Make a new subgraph
      tflite::Subgraph* new_subgraph = interpreter_->CreateSubgraphInLevel(level);
      subgraphs_created.push_back(new_subgraph);
      if(!prev_queue.empty()){ // make linked-list structure of subgraphs
        prev_queue.front()->SetNextSubgraph(new_subgraph);
        new_subgraph->SetPrevSubgraph(prev_queue.front());
        prev_queue.pop();
      }
      prev_queue.push(new_subgraph);
      const int* nodes_in_partition = master_partitioning_plan[partition_itr]->nodes;
      const int num_nodes_in_partition = master_partitioning_plan[partition_itr]->size;
      std::cout << "partitioning ratio at iteration " << 
                master_partitioning_plan[partition_itr]->partitioning_ratios[0] << "\n";
      switch (master_partitioning_plan[partition_itr]->resource_type)
      {
      case ResourceType::CPU:
        // Set this sugraph for cpu subgraph
        new_subgraph->SetResourceType(ResourceType::CPU);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);
        new_subgraph->context()->recommended_num_threads = 6;
        break;
      case ResourceType::GPU:
        // Set this sugraph for gpu subgraph
        new_subgraph->SetResourceType(ResourceType::GPU);
        new_subgraph->context()->recommended_num_threads = 6;  
        break;
      case ResourceType::CO_CPU:
        new_subgraph->SetResourceType(ResourceType::CO_CPU);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);  
        if(master_partitioning_plan[partition_itr]->partitioning_ratios[0] > 10)
          new_subgraph->SetPartitioningType(PartitioningType::HEIGHT_PARTITIONING);
        else
          new_subgraph->SetPartitioningType(PartitioningType::CHANNEL_PARTITIONING);
        new_subgraph->context()->recommended_num_threads = 6;
        break;
      case ResourceType::CO_GPU:
        new_subgraph->SetResourceType(ResourceType::CO_GPU);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);
        if(master_partitioning_plan[partition_itr]->partitioning_ratios[0] > 10)
          new_subgraph->SetPartitioningType(PartitioningType::HEIGHT_PARTITIONING);
        else
          new_subgraph->SetPartitioningType(PartitioningType::CHANNEL_PARTITIONING);        
          new_subgraph->context()->recommended_num_threads = 6;  
        break;
      case ResourceType::CPU_XNN:
        new_subgraph->SetResourceType(ResourceType::CPU_XNN);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);
        new_subgraph->context()->recommended_num_threads = 6;
        break;
      case ResourceType::CO_CPU_XNN:
        new_subgraph->SetResourceType(ResourceType::CO_CPU_XNN);
        new_subgraph->PushExternalParameter(
          master_partitioning_plan[partition_itr]->partitioning_ratios[0]);
        if(master_partitioning_plan[partition_itr]->partitioning_ratios[0] > 10)
          new_subgraph->SetPartitioningType(PartitioningType::HEIGHT_PARTITIONING);
        else
          new_subgraph->SetPartitioningType(PartitioningType::CHANNEL_PARTITIONING);        
        break;
      default:
        break;
      }
      std::vector<int> nodes_to_parse;
      // Now setup nodes and tensors for new subgraph
      std::cout << "partitioning_iteration : " << partition_itr << "\n";  
      for(int j=0; j < num_nodes_in_partition; ++j){
        int working_op = nodes_in_partition[j];
        nodes_to_parse.push_back(working_op);
        const auto* op = operators->Get(working_op);
        int op_index = op->opcode_index();
        /// get every tensor indices of all nodes
        for(int k=0; k<FlatBufferIntArrayToVector(op->inputs()).size(); ++k)
          tensors_->push_back(FlatBufferIntArrayToVector(op->inputs())[k]);
        for(int k=0; k<FlatBufferIntArrayToVector(op->outputs()).size(); ++k)
          tensors_->push_back(FlatBufferIntArrayToVector(op->outputs())[k]);          
        /// input tensor should be first node's input tensor in partitioning plan
        if(j == 0){
          input_tensor = new std::vector<int>;
          input_tensor->push_back(FlatBufferIntArrayToVector(op->inputs())[0]);
          new_subgraph->SetActualInput(*input_tensor); // set 'actual' input tensors
        }  /// output tensor should be last node's output tensor in partitioning plan
        if(j == num_nodes_in_partition - 1){ // last iteration
          output_tensor = new std::vector<int>;
          output_tensor->push_back(FlatBufferIntArrayToVector(op->outputs())[0]);
          new_subgraph->SetActualOutput(*output_tensor); // set 'actual' output tensors
          if (new_subgraph->AddTensors(tensors->size()) != kTfLiteOk){
            return kTfLiteError;
          }
          std::pair<int, std::vector<int>> graph_and_tensors(partition_itr, *tensors_);
          subgraph_and_tensors.push_back(graph_and_tensors);
          new_subgraph->SetInputs(  // set 'all' input tensors
                      std::vector<int>(input_tensor->begin(), input_tensor->end()));
          new_subgraph->SetOutputs(  // set 'all' output tensors
                      std::vector<int>(output_tensor->begin(), output_tensor->end()));
          if (ParseNodes(operators, new_subgraph, nodes_to_parse) != kTfLiteOk)
            return kTfLiteError;
          if (ParseTensors(buffers, tensors, new_subgraph, *tensors_) != kTfLiteOk)
            return kTfLiteError;
          std::vector<int> variables;
          for (int l = 0; l < new_subgraph->tensors_size(); ++l) {
            auto* tensor = new_subgraph->tensor(l);
            if (tensor->is_variable) {
              variables.push_back(l);
            }
          }
          new_subgraph->SetVariables(std::move(variables));
          std::cout << "Interpreterbuilder : created new subgraph" << "\n";
        }
      }
      input_tensor->clear();
      delete input_tensor;
      tensors_->clear();
      delete tensors_;
      output_tensor->clear();
      delete output_tensor;
      tensors_ = new std::vector<int>;
      output_tensor = new std::vector<int>;
    }
    std::cout << "RegisterSubgraphToInterpreter" << "\n";
    if(RegisterSubgraphToInterpreter(level ,subgraphs_created) != kTfLiteOk){
      std::cout << "RegisterSubgraphToInterpreter ERROR" << "\n";
      return kTfLiteError;
    }
    interpreter_->PrintSubgraphInfo(); 
    // Fill shared tensor bucket  
    for(size_t graph_idx=0; graph_idx<subgraph_and_tensors.size(); ++graph_idx){
      for(size_t j=0; j<subgraph_and_tensors[graph_idx].second.size(); ++j){
        int tensor = subgraph_and_tensors[graph_idx].second[j];
        if(shared_tensor_bucket[tensor][graph_idx] != 1)
          shared_tensor_bucket[tensor][graph_idx] = 1; // Tensor shared flag
      }
    }
    // Save shared intermediate tensor indices in interpreter's graph_and_shared_tensor.
    // Used when AllocateTensorsofAllSubgraphs called.
    // (to propagate tensor shapes)™
    for(size_t t=0; t<tensors->size(); ++t){
      std::pair<int, std::vector<int>> pair_tensor_graph;
      std::vector<int> sharing_subgraph_id;
      for(size_t g=0; g<subgraph_and_tensors.size(); ++g){
        if(shared_tensor_bucket[t][g]){
          sharing_subgraph_id.push_back(subgraphs_created[g]->GetGraphid());
        }
      }
      if(sharing_subgraph_id.size() > 1){
        pair_tensor_graph.first = t;        // tensor index
        pair_tensor_graph.second = sharing_subgraph_id; // subgraph id
        shared_info.push_back(pair_tensor_graph);
      }
      sharing_subgraph_id.clear();
    }
    SharedTensorsInGraphs* temp = new SharedTensorsInGraphs;
    temp->pair_tensor_graph = shared_info;
    temp->model_id = model_id_;
    if((interpreter_)->shared_tensor_and_graph.size() <= level){
      (interpreter_)->shared_tensor_and_graph.push_back(std::vector<SharedTensorsInGraphs*>());
    }
    (interpreter_)->shared_tensor_and_graph[level].push_back(temp);
  }
  
  std::vector<int> subgraph_set;
  interpreter_->GetTotalSubgraphID(subgraph_set);
  
  // Delete old subgraph and move it to interpreter's primary subgraph.
  if(profiled_subgraph != nullptr){
    if(interpreter_->DeleteSubgraph(profiled_subgraph->GetGraphid()) 
        != kTfLiteOk){
      std::cout << "DeleteSubgraph ERROR" << "\n";
      return kTfLiteError;
    }
  }
  interpreter_->PrintSubgraphInfo(); 
  // std::vector<int> subgraph_set;
  interpreter_->GetTotalSubgraphID(subgraph_set);

  // MUST CHECK
  // Does CPU-side interpreter need to call AllocateTensors twice?
  if(interpreter_->AllocateTensorsofSubsets(level, model_id_) != kTfLiteOk){
    std::cout << "AllocateTensorsofSubsets ERROR" << "\n";
    return kTfLiteError;
  }
  std::cout << "Allocated tensors" << "\n";
  if(is_sub_interpreter){ // If Co-execution CPU InterpreterBuilder
    if(PartitionChannels((subgraphs_created)) != kTfLiteOk){
      std::cout << "Partition channel-wise for cpu ERROR" << "\n";
      return kTfLiteError;
    }
  }
  if(DelegateSubgraphs(subgraphs_created) != kTfLiteOk){
    std::cout << "DelegateOldSubgraphs ERROR" << "\n";
    return kTfLiteError;
  }
  std::cout << "Delegate tensors" << "\n";
  std::cout << "Interpreterbuilder: Subgraphs created" << "\n";
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::DelegateSubgraphs(
                    std::vector<tflite::Subgraph*>& new_subgraphs){
  std::cout << "Delegate " << new_subgraphs.size() << " subgraphs" << "\n";
  for(auto new_subgraph : new_subgraphs){
    // sj
    // To check whether subgraphs need delegation
    double response_time = 0;
    struct timespec begin, end; 
    clock_gettime(CLOCK_MONOTONIC, &begin);
    std::cout << "resource type : " << new_subgraph->GetResourceType() << "\n";
    if(new_subgraph->GetResourceType() == ResourceType::GPU ||
      new_subgraph->GetResourceType() == ResourceType::CO_GPU ||
      new_subgraph->GetResourceType() == ResourceType::CPU_XNN ||
      new_subgraph->GetResourceType() == ResourceType::CO_CPU_XNN){
      // Minsung
      // Experimental flag for ADD, MUL fallback handle in YOLO.
      if(interpreter_->GetInputType() == INPUT_TYPE::COCO416){
        new_subgraph->SetExperimentalFlagTrue();
        std::cout << "SetExperimentalFlagTrue" << "\n";
      }
      else{
        new_subgraph->SetExperimentalFlagFalse();
        std::cout << "SetExperimentalFlagFalse" << "\n";
      }

      if(interpreter_->ModifyGraphWithDelegateImpl(new_subgraph->GetGraphid())
        != kTfLiteOk){
          std::cout << "Graph ID " << new_subgraph->GetGraphid() << "Failed to"
                  << " Delegate" << "\n";
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      printf("Delegate %.6f \n", response_time);
      response_time = (end.tv_sec - begin.tv_sec) +
                ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::PartitionChannels(
                    std::vector<tflite::Subgraph*>& new_subgraphs){
  for(auto new_subgraph : new_subgraphs){
   if(new_subgraph->GetResourceType() == ResourceType::CO_CPU ||
        // sj
        // Fixed for CO_CPU_XNN.. (785a4) MINSUNG
        new_subgraph->GetResourceType() == ResourceType::CO_CPU_XNN){
      if(new_subgraph->PartitionChannel() != kTfLiteOk){
          std::cout << "Graph ID " << new_subgraph->GetGraphid() << "Failed to"
                   << " Partition in channel-wise" << "\n";
          return kTfLiteError;
      }
    }
  }
  return kTfLiteOk;                  
}


TfLiteStatus InterpreterBuilder::RegisterSubgraphToInterpreter(
                                int level,
                                std::vector<tflite::Subgraph*>& new_subgraphs){
  for(size_t i=0; i<new_subgraphs.size(); ++i){
    // give model id
    new_subgraphs[i]->SetModelid(model_id_);
    // give subgraph id
    new_subgraphs[i]->SetGraphid(interpreter_->GetAndAddSubgraphIDCreated());
    // give level info
    new_subgraphs[i]->SetLevel(level);
    // interpreter->AddNewSubgraph
    if(interpreter_->AddNewSubgraph(level, new_subgraphs[i]) != kTfLiteOk){
      std::cout << "AddNewSubgraph ERROR" << "\n";
      return kTfLiteError;
    }
    if(interpreter_->RegisterSubgraphSubsets(level, new_subgraphs[i]) != kTfLiteOk){
      std::cout << "RegisterSubgraph ERROR" << "\n";
      return kTfLiteError;
    }
    std::cout << "Registered new subgraph of id [" << new_subgraphs[i]->GetGraphid() << "]" 
              << " in level [" << level << "]\n"; 
  }
  // In case if there is no subgraph to register.
  // We must initialize subgraphs__[level] with empty vector anyway.
  if(new_subgraphs.empty()){
    interpreter_->AddNewSubgraph(level, nullptr);
  }
  return kTfLiteOk;
}


TfLiteStatus InterpreterBuilder::ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Subgraph* subgraph) {
  TfLiteStatus status = kTfLiteOk;

  // Reduce the number of redundant allocations
  subgraph->ReserveNodes(operators->size());

  for (int i = 0; i < operators->size(); ++i) {
    const auto* op = operators->Get(i);
    int index = op->opcode_index();
    if (index < 0 || index >= flatbuffer_op_index_to_registration_.size()) {
      error_reporter_->Report("Missing registration for opcode_index %d\n",
                              index);
      status = kTfLiteError;
      continue;
    }

    const TfLiteRegistration* registration =
        flatbuffer_op_index_to_registration_[index];
    if (registration == nullptr) {
      error_reporter_->Report("Skipping op for opcode_index %d\n", index);
      status = kTfLiteError;
      continue;
    }

    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);

    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      error_reporter_->Report(
          "Found builtin operator %s with custom options.\n",
          EnumNameBuiltinOperator(op_type));
    }

    if (op_type == BuiltinOperator_CUSTOM) {
      if (op->custom_options()) {
        subgraph->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            FlatBufferIntArrayToVector(op->intermediates()),
            reinterpret_cast<const char*>(op->custom_options()->data()),
            op->custom_options()->size(), nullptr, registration);
      } else {
        subgraph->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            FlatBufferIntArrayToVector(op->intermediates()), nullptr, 0,
            nullptr, registration);
      }
    } else {
      void* builtin_data = nullptr;
      MallocDataAllocator malloc_allocator;
      TF_LITE_ENSURE_STATUS(ParseOpData(op, op_type, error_reporter_,
                                        &malloc_allocator, &builtin_data));
      subgraph->AddNodeWithParameters(
          FlatBufferIntArrayToVector(op->inputs()),
          FlatBufferIntArrayToVector(op->outputs()),
          FlatBufferIntArrayToVector(op->intermediates()), nullptr, 0,
          builtin_data, registration);
    }
  }

  return status;
}



TfLiteStatus InterpreterBuilder::ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Subgraph* subgraph, int op_st, int op_end) {
  TfLiteStatus status = kTfLiteOk;

  // Reduce the number of redundant allocations
  subgraph->ReserveNodes(op_end);
  //if(op_st == op_end) // If a Layer has single node
  op_end++;
  for (int i = op_st; i < op_end; ++i) {
    const auto* op = operators->Get(i);
    int index = op->opcode_index();
    if (index < 0 || index >= flatbuffer_op_index_to_registration_.size()) {
      error_reporter_->Report("Missing registration for opcode_index %d\n",
                              index);
      status = kTfLiteError;
      continue;
    }
    const TfLiteRegistration* registration =
        flatbuffer_op_index_to_registration_[index];
    if (registration == nullptr) {
      error_reporter_->Report("Skipping op for opcode_index %d\n", index);
      status = kTfLiteError;
      continue;
    }

    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);
    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      error_reporter_->Report(
          "Found builtin operator %s with custom options.\n",
          EnumNameBuiltinOperator(op_type));
    }

    if (op_type == BuiltinOperator_CUSTOM) {
      if (op->custom_options()) {
        subgraph->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            FlatBufferIntArrayToVector(op->intermediates()),
            reinterpret_cast<const char*>(op->custom_options()->data()),
            op->custom_options()->size(), nullptr, registration);
      } else {
        subgraph->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            FlatBufferIntArrayToVector(op->intermediates()), nullptr, 0,
            nullptr, registration);
      }
    } else {
      void* builtin_data = nullptr;
      MallocDataAllocator malloc_allocator;
      TF_LITE_ENSURE_STATUS(ParseOpData(op, op_type, error_reporter_,
                                        &malloc_allocator, &builtin_data));
      subgraph->AddNodeWithParameters(
          FlatBufferIntArrayToVector(op->inputs()),
          FlatBufferIntArrayToVector(op->outputs()),
          FlatBufferIntArrayToVector(op->intermediates()), nullptr, 0,
          builtin_data, registration);
    }
  }
  return status;
}

TfLiteStatus InterpreterBuilder::ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Subgraph* subgraph, std::vector<int>& nodes_to_parse) {
  TfLiteStatus status = kTfLiteOk;

  // Reduce the number of redundant allocations
  if(nodes_to_parse.empty()){
    std::cout << "InterpreterBuilder ERROR" << "\n";
    std::cout << "Threre's no node to parse" << "\n";
    status = kTfLiteError;
  }
  subgraph->ReserveNodes(nodes_to_parse.size());

  for(int idx=0; idx<nodes_to_parse.size(); ++ idx){
    int op_idx = nodes_to_parse[idx];
    const auto* op = operators->Get(op_idx);
    int opcode_idx = op->opcode_index();
    if (opcode_idx < 0 || opcode_idx >= flatbuffer_op_index_to_registration_.size()) {
      error_reporter_->Report("Missing registration for opcode_index %d\n",
                              opcode_idx);
      status = kTfLiteError;
      continue;
    }
    const TfLiteRegistration* registration =
        flatbuffer_op_index_to_registration_[opcode_idx];
    if (registration == nullptr) {
      error_reporter_->Report("Skipping op for opcode_index %d\n", opcode_idx);
      status = kTfLiteError;
      continue;
    }

    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);
    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      error_reporter_->Report(
          "Found builtin operator %s with custom options.\n",
          EnumNameBuiltinOperator(op_type));
    }
    if (op_type == BuiltinOperator_CUSTOM) {
      if (op->custom_options()) {
        subgraph->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            FlatBufferIntArrayToVector(op->intermediates()),
            reinterpret_cast<const char*>(op->custom_options()->data()),
            op->custom_options()->size(), nullptr, registration);
      } else {
        subgraph->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            FlatBufferIntArrayToVector(op->intermediates()), nullptr, 0,
            nullptr, registration);
      }
    } else {
      void* builtin_data = nullptr;
      MallocDataAllocator malloc_allocator;
      TF_LITE_ENSURE_STATUS(ParseOpData(op, op_type, error_reporter_,
                                        &malloc_allocator, &builtin_data));
      subgraph->AddNodeWithParameters(
          FlatBufferIntArrayToVector(op->inputs()),
          FlatBufferIntArrayToVector(op->outputs()),
          FlatBufferIntArrayToVector(op->intermediates()), nullptr, 0,
          builtin_data, registration);
    }
  }
  return status;
}


// Minsung
// ParseTensors with specific index of operators
TfLiteStatus InterpreterBuilder::ParseTensors(
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
    Subgraph* subgraph, std::vector<int> tensor_idx) {
  TfLiteStatus status = kTfLiteOk;
  // A little helper to get the names of inputs and outputs. Note that they
  // must outlive the subgraph.
  auto get_name = [](const tflite::Tensor* t) -> const char* {
    auto name = t->name();
    if (name) return name->c_str();
    return kEmptyTensorName;
  };
  num_fp32_tensors_ = 0;
  for (int i = 0; i < tensor_idx.size(); ++i) {
    const auto* tensor = tensors->Get(tensor_idx[i]);
    std::vector<int> dims = FlatBufferIntArrayToVector(tensor->shape());
    TfLiteType type;
    if (ConvertTensorType(tensor->type(), &type, error_reporter_) !=
        kTfLiteOk) {
      status = kTfLiteError;
      continue;
    }
    if (type == kTfLiteFloat32) {
      ++num_fp32_tensors_;
    }
    auto get_readonly_data = [&](const char** buffer_data,
                                 size_t* buffer_size) {
      // TODO(aselle): Check what happens if we have an unspecified size
      // constant.
      *buffer_data = nullptr;
      if (tensor->buffer() == 0) return kTfLiteOk;
      if (tensor->buffer() >= buffers->size()) {
        error_reporter_->Report(
            "Tensor %d specifies out of range buffer %d (only %d buffers).\n",
            tensor_idx[i], tensor->buffer(), buffers->size());
        return kTfLiteError;
      }
      if (auto* buffer = (*buffers)[tensor->buffer()]) {
        if (auto* array = buffer->data()) {
          if (size_t size = array->size()) {
            *buffer_size = size;
            *buffer_data = reinterpret_cast<const char*>(array->data());
            return kTfLiteOk;
          }
        }
      }
      return kTfLiteOk;
    };
    size_t buffer_size = 0;
    const char* buffer_ptr;
    TF_LITE_ENSURE_STATUS(get_readonly_data(&buffer_ptr, &buffer_size));

    const auto* src_quantization = tensor->quantization();
    TfLiteQuantization quantization;
    if (ParseQuantization(src_quantization, &quantization, dims) != kTfLiteOk) {
      error_reporter_->Report("Tensor %d has invalid quantization parameters.",
                              tensor_idx[i]);
      status = kTfLiteError;
    }
    size_t dims_signature_rank = 0;
    const int* dims_signature_data = nullptr;
    if (tensor->shape_signature()) {
      dims_signature_rank = tensor->shape_signature()->size();
      dims_signature_data = tensor->shape_signature()->data();
    }

    bool is_variable = tensor->is_variable();
    if (buffer_ptr) {
      if (is_variable) {
        error_reporter_->Report(
            "Tensor %d is a variable tensor with buffer. "
            "It's not supported now.\n",
            i);
        status = kTfLiteError;
      }

      // TODO(b/144999664): Only constant sparse tensor is supported now.
      const auto* src_sparsity = tensor->sparsity();
      TfLiteSparsity* sparsity = nullptr;
      if (ParseSparsity(src_sparsity, &sparsity) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d has invalid sparsity parameters.",
                                tensor_idx[i]);
        status = kTfLiteError;
      }
      if (subgraph->SetTensorParametersReadOnly(
              tensor_idx[i], type, get_name(tensor), dims, quantization, buffer_ptr,
              buffer_size, allocation_, sparsity) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                tensor_idx[i]);
        status = kTfLiteError;
      }
    } else {
      if (subgraph->SetTensorParametersReadWrite(
              tensor_idx[i], type, get_name(tensor), dims, quantization, is_variable,
              dims_signature_rank, dims_signature_data) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                tensor_idx[i]);
        status = kTfLiteError;
      }
    }
  }
  
  return status;
}


TfLiteStatus InterpreterBuilder::ParseQuantization(
    const QuantizationParameters* src_quantization,
    TfLiteQuantization* quantization, const std::vector<int>& dims) {
  quantization->type = kTfLiteNoQuantization;
  if (!src_quantization || !src_quantization->scale() ||
      src_quantization->scale()->size() == 0) {
    return kTfLiteOk;
  }
  if (!src_quantization->zero_point()) {
    error_reporter_->Report(
        "Quantization parameters has non-null scale but null zero_point.");
    return kTfLiteError;
  }

  // Ensure that the number of scales matches the number of zero_points.
  if (src_quantization->scale()->size() !=
      src_quantization->zero_point()->size()) {
    error_reporter_->Report(
        "QuantizationParam has %d zero_point values and %d scale values. Must "
        "have same number.",
        src_quantization->zero_point()->size(),
        src_quantization->scale()->size());
    return kTfLiteError;
  }

  const size_t num_scales = src_quantization->scale()->size();

  // Ensure that the quantization dimension is valid.
  if (src_quantization->quantized_dimension() < 0 ||
      (!dims.empty() &&
       src_quantization->quantized_dimension() >= dims.size())) {
    error_reporter_->Report(
        "quantized_dimension must be in range [0, %d). Was %d.", dims.size(),
        src_quantization->quantized_dimension());
    return kTfLiteError;
  }

  // Ensure that the number of scales is 1 for per-layer quantization, and
  // matches number of quantization dimensions for per-axis quantization.
  if (num_scales != 1 &&
      (!dims.empty() &&
       num_scales != dims[src_quantization->quantized_dimension()])) {
    error_reporter_->Report(
        "num_scales must be 1 for per-layer quantization, or %d for per-axis "
        "quantization, but got %d.",
        dims[src_quantization->quantized_dimension()], num_scales);
    return kTfLiteError;
  }

  // Affine-quantization.
  quantization->type = kTfLiteAffineQuantization;
  auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  affine_quantization->scale = TfLiteFloatArrayCreate(num_scales);
  affine_quantization->zero_point = TfLiteIntArrayCreate(num_scales);
  for (size_t i = 0; i < num_scales; ++i) {
    affine_quantization->scale->data[i] = src_quantization->scale()->Get(i);
    affine_quantization->zero_point->data[i] =
        src_quantization->zero_point()->Get(i);
  }
  affine_quantization->quantized_dimension =
      src_quantization->quantized_dimension();
  quantization->params = reinterpret_cast<void*>(affine_quantization);
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::ParseSparsity(
    const SparsityParameters* src_sparsity, TfLiteSparsity** sparsity_ptr) {
  if (!src_sparsity) {
    return kTfLiteOk;
  }

  if (src_sparsity->traversal_order() == nullptr ||
      src_sparsity->dim_metadata() == nullptr) {
    error_reporter_->Report("Invalid sparsity parameter.");
    return kTfLiteError;
  }

  auto* sparsity =
      reinterpret_cast<TfLiteSparsity*>(malloc(sizeof(TfLiteSparsity)));
  memset(sparsity, 0, sizeof(TfLiteSparsity));
  *sparsity_ptr = sparsity;

  const size_t traversal_order_size = src_sparsity->traversal_order()->size();
  sparsity->traversal_order = TfLiteIntArrayCreate(traversal_order_size);
  for (int i = 0; i < traversal_order_size; i++) {
    sparsity->traversal_order->data[i] =
        src_sparsity->traversal_order()->Get(i);
  }

  if (src_sparsity->block_map()) {
    const size_t block_map_size = src_sparsity->block_map()->size();
    sparsity->block_map = TfLiteIntArrayCreate(block_map_size);
    for (int i = 0; i < block_map_size; i++) {
      sparsity->block_map->data[i] = src_sparsity->block_map()->Get(i);
    }
  }

  const size_t dim_metadata_size = src_sparsity->dim_metadata()->size();
  sparsity->dim_metadata_size = dim_metadata_size;
  sparsity->dim_metadata = reinterpret_cast<TfLiteDimensionMetadata*>(
      malloc(dim_metadata_size * sizeof(TfLiteDimensionMetadata)));
  memset(sparsity->dim_metadata, 0,
         dim_metadata_size * sizeof(TfLiteDimensionMetadata));

  for (int i = 0; i < dim_metadata_size; i++) {
    const auto* src_metadata = src_sparsity->dim_metadata()->Get(i);
    if (src_metadata->format() != DimensionType_DENSE &&
        src_metadata->format() != DimensionType_SPARSE_CSR) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "The %dth dimension has unknown type: %d.", i,
                           src_metadata->format());
      return kTfLiteError;
    }
    auto* tgt_metadata = &sparsity->dim_metadata[i];

    tgt_metadata->format =
        static_cast<TfLiteDimensionType>(src_metadata->format());

    if (tgt_metadata->format == kTfLiteDimDense) {
      tgt_metadata->dense_size = src_metadata->dense_size();
    } else {
      if (ParseSparseIndexVector(src_metadata, tgt_metadata) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "The %dth sparse dimension has invalid parameters.", i);
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::ParseTensors(
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
    Subgraph* subgraph) {
  TfLiteStatus status = kTfLiteOk;

  // A little helper to get the names of inputs and outputs. Note that they
  // must outlive the subgraph.
  auto get_name = [](const tflite::Tensor* t) -> const char* {
    auto name = t->name();
    if (name) return name->c_str();
    return kEmptyTensorName;
  };

  num_fp32_tensors_ = 0;
  for (int i = 0; i < tensors->size(); ++i) {
    const auto* tensor = tensors->Get(i);
    std::vector<int> dims = FlatBufferIntArrayToVector(tensor->shape());

    TfLiteType type;
    if (ConvertTensorType(tensor->type(), &type, error_reporter_) !=
        kTfLiteOk) {
      status = kTfLiteError;
      continue;
    }
    if (type == kTfLiteFloat32) {
      ++num_fp32_tensors_;
    }
    auto get_readonly_data = [&](const char** buffer_data,
                                 size_t* buffer_size) {
      // TODO(aselle): Check what happens if we have an unspecified size
      // constant.
      *buffer_data = nullptr;
      if (tensor->buffer() == 0) return kTfLiteOk;
      if (tensor->buffer() >= buffers->size()) {
        error_reporter_->Report(
            "Tensor %d specifies out of range buffer %d (only %d buffers).\n",
            i, tensor->buffer(), buffers->size());
        return kTfLiteError;
      }
      if (auto* buffer = (*buffers)[tensor->buffer()]) {
        if (auto* array = buffer->data()) {
          if (size_t size = array->size()) {
            *buffer_size = size;
            *buffer_data = reinterpret_cast<const char*>(array->data());
            return kTfLiteOk;
          }
        }
      }
      return kTfLiteOk;
    };
    size_t buffer_size = 0;
    const char* buffer_ptr;
    TF_LITE_ENSURE_STATUS(get_readonly_data(&buffer_ptr, &buffer_size));

    const auto* src_quantization = tensor->quantization();
    TfLiteQuantization quantization;
    if (ParseQuantization(src_quantization, &quantization, dims) != kTfLiteOk) {
      error_reporter_->Report("Tensor %d has invalid quantization parameters.",
                              i);
      status = kTfLiteError;
    }

    size_t dims_signature_rank = 0;
    const int* dims_signature_data = nullptr;
    if (tensor->shape_signature()) {
      dims_signature_rank = tensor->shape_signature()->size();
      dims_signature_data = tensor->shape_signature()->data();
    }

    bool is_variable = tensor->is_variable();
    if (buffer_ptr) {
      if (is_variable) {
        error_reporter_->Report(
            "Tensor %d is a variable tensor with buffer. "
            "It's not supported now.\n",
            i);
        status = kTfLiteError;
      }

      // TODO(b/144999664): Only constant sparse tensor is supported now.
      const auto* src_sparsity = tensor->sparsity();
      TfLiteSparsity* sparsity = nullptr;
      if (ParseSparsity(src_sparsity, &sparsity) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d has invalid sparsity parameters.",
                                i);
        status = kTfLiteError;
      }

      if (subgraph->SetTensorParametersReadOnly(
              i, type, get_name(tensor), dims, quantization, buffer_ptr,
              buffer_size, allocation_, sparsity) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                i);
        status = kTfLiteError;
      }
    } else {
      if (subgraph->SetTensorParametersReadWrite(
              i, type, get_name(tensor), dims, quantization, is_variable,
              dims_signature_rank, dims_signature_data) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                i);
        status = kTfLiteError;
      }
    }
  }

  return status;
}

TfLiteStatus InterpreterBuilder::ApplyDelegates(Interpreter* interpreter,
                                                int num_threads) {
  // Apply Flex delegate if applicable.
  if (has_flex_op_) {
    if (auto flex_delegate = AcquireFlexDelegate()) {
      return interpreter->ModifyGraphWithDelegate(std::move(flex_delegate));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::operator()(
    std::unique_ptr<Interpreter>* interpreter) {
  return operator()(interpreter, /*num_threads=*/-1);
}

TfLiteStatus InterpreterBuilder::operator()(
    std::unique_ptr<Interpreter>* interpreter, int num_threads) {
  std::cout << "Interpreterbuilder creates subgraph" << "\n";
  if (!interpreter) {
    error_reporter_->Report(
        "Null output pointer passed to InterpreterBuilder.");
    return kTfLiteError;
  }

  if (num_threads < -1) {
    error_reporter_->Report(
        "num_threads should be >=0 or just -1 to let TFLite runtime set the "
        "value.");
    return kTfLiteError;
  }

  // Safe exit by deleting partially created interpreter, to reduce verbosity
  // on error conditions. Use by return cleanup_on_error();
  auto cleanup_and_error = [&interpreter]() {
    interpreter->reset();
    return kTfLiteError;
  };

  if (!model_) {
    error_reporter_->Report("Null pointer passed in as model.");
    return cleanup_and_error();
  }

  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter_->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model_->version(), TFLITE_SCHEMA_VERSION);
    return cleanup_and_error();
  }

  if (BuildLocalIndexToRegistrationMapping() != kTfLiteOk) {
    error_reporter_->Report("Registration failed.\n");
    return cleanup_and_error();
  }

  // Flatbuffer model schemas define a list of opcodes independent of the graph.
  // We first map those to registrations. This reduces string lookups for custom
  // ops since we only do it once per custom op rather than once per custom op
  // invocation in the model graph.
  // Construct interpreter with correct number of tensors and operators.
  auto* subgraphs = model_->subgraphs();
  auto* buffers = model_->buffers();

  if (subgraphs->size() == 0) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No subgraph in the model.\n");
    return cleanup_and_error();
  }

  if (!buffers) {
    TF_LITE_REPORT_ERROR(error_reporter_, "No buffers in the model.\n");
    return cleanup_and_error();
  }

  interpreter->reset(new Interpreter(error_reporter_));
  (*interpreter)->SetNumThreads(num_threads);
  if (subgraphs->size() > 1) {
    (*interpreter)->AddSubgraphs(subgraphs->size() - 1);
  }

  (*interpreter)->SetProfiler(tflite::profiling::MaybeCreatePlatformProfiler());

  for (int subgraph_index = 0; subgraph_index < subgraphs->size();
       ++subgraph_index) {
    const tflite::SubGraph* subgraph = (*subgraphs)[subgraph_index];
    tflite::Subgraph* modified_subgraph =
        (*interpreter)->subgraph(subgraph_index);
    auto operators = subgraph->operators();
    auto tensors = subgraph->tensors();
    if (!operators || !tensors) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Did not get operators or tensors in subgraph %d.\n",
                           subgraph_index);
      return cleanup_and_error();
    }
    if (modified_subgraph->AddTensors(tensors->size()) != kTfLiteOk) {
      return cleanup_and_error();
    }
    // Set num threads
    // Parse inputs/outputs
    modified_subgraph->SetInputs(
        FlatBufferIntArrayToVector(subgraph->inputs()));
    modified_subgraph->SetOutputs(
        FlatBufferIntArrayToVector(subgraph->outputs()));

    // Finally setup nodes and tensors
    if (ParseNodes(operators, modified_subgraph) != kTfLiteOk)
      return cleanup_and_error();
    if (ParseTensors(buffers, tensors, modified_subgraph) != kTfLiteOk)
      return cleanup_and_error();

    std::vector<int> variables;
    for (int i = 0; i < modified_subgraph->tensors_size(); ++i) {
      auto* tensor = modified_subgraph->tensor(i);
      if (tensor->is_variable) {
        variables.push_back(i);
      }
    }
    modified_subgraph->SetVariables(std::move(variables));
  }

  if (num_fp32_tensors_ > 0) {
    (*interpreter)->lazy_delegate_providers_ =
        op_resolver_.GetDelegates(num_threads);
  }

  if (ApplyDelegates(interpreter_, num_threads) != kTfLiteOk)
    return cleanup_and_error();
  std::cout << "Interpreterbuilder creates subgraph done" << "\n";
  return kTfLiteOk;
}

}  // namespace tflite

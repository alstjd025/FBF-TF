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
/// \file
/// Provides functionality to construct an interpreter for a model.
///
#ifndef TENSORFLOW_LITE_INTERPRETER_BUILDER_H_
#define TENSORFLOW_LITE_INTERPRETER_BUILDER_H_


#include <memory>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <iostream>

namespace tflite {

/// Build an interpreter capable of interpreting `model`.
///
/// `model`: A model whose lifetime must be at least as long as any
///   interpreter(s) created by the builder. In principle multiple interpreters
///   can be made from a single model.
/// `op_resolver`: An instance that implements the `OpResolver` interface, which
///   maps custom op names and builtin op codes to op registrations. The
///   lifetime of the provided `op_resolver` object must be at least as long as
///   the `InterpreterBuilder`; unlike `model` and `error_reporter`, the
///   `op_resolver` does not need to exist for the duration of any created
///   `Interpreter` objects.
/// `error_reporter`: a functor that is called to report errors that handles
///   printf var arg semantics. The lifetime of the `error_reporter` object must
///   be greater than or equal to the `Interpreter` created by `operator()`.
///
/// Returns a kTfLiteOk when successful and sets interpreter to a valid
/// Interpreter. Note: The user must ensure the lifetime of the model (and error
/// reporter, if provided) is at least as long as interpreter's lifetime.
class InterpreterBuilder {
 public:

  InterpreterBuilder();
    
  InterpreterBuilder(const FlatBufferModel& model,
                     const OpResolver& op_resolver);
  /// Builds an interpreter given only the raw flatbuffer Model object (instead
  /// of a FlatBufferModel). Mostly used for testing.
  /// If `error_reporter` is null, then DefaultErrorReporter() is used.
  InterpreterBuilder(const ::tflite::Model* model,
                     const OpResolver& op_resolver,
                     ErrorReporter* error_reporter = DefaultErrorReporter());

  // Minsung
  // Use this constructor
  InterpreterBuilder(const FlatBufferModel& model,
                    const OpResolver& op_resolver,
                    const char* model_name,
                    int model_id);

  // Minsung
  // Use this constructor for dummy profile
  InterpreterBuilder(const FlatBufferModel& model,
                    const OpResolver& op_resolver,
                    Interpreter* interpreter,
                    const char* model_name,
                    int model_id, bool is_co_execution_cpu);

  ~InterpreterBuilder();
  InterpreterBuilder(const InterpreterBuilder&) = delete;
  InterpreterBuilder& operator=(const InterpreterBuilder&) = delete;
  TfLiteStatus operator()(std::unique_ptr<Interpreter>* interpreter);
  TfLiteStatus operator()(std::unique_ptr<Interpreter>* interpreter,
                          int num_threads);

  int GetModelid() { return model_id_; }
  const std::string GetModelName() { return model_name_; }
  Interpreter* GetInterpreter() { return interpreter_; }
  // Minsung
  // Creates a subgraph.
  // This funciton is an initial phase of creating a model into invokable subgraph and
  // move it to the given interpreter.
  // and wrap it with default job class
  // (parse nodes, tensors and allocatetensors)
  TfLiteStatus CreateSubgraphFromFlatBuffer();

  // Minsung
  // Creates subset of subgraphs
  // After profiling the whole subgraph's latency, creates subset of subgraphs so
  // that the scheduler can handle them.
  TfLiteStatus CreateSubgraphsFromParameter(tflite::Subgraph* profiled_subgraph);

  // Minsung
  // Creates multi-level subgraphs level by level.

  TfLiteStatus CreateSubgraphsFromParameter(int level, tflite::Subgraph* profiled_subgraph);

  // Minsung
  // Creates a subgraph for stress test.
  TfLiteStatus CreateSubgraphForStress();

  TfLiteStatus DelegateSubgraphs(std::vector<tflite::Subgraph*>& new_subgraphs);

  TfLiteStatus RegisterSubgraphToInterpreter(
                                      int level, 
                                      std::vector<tflite::Subgraph*>& new_subgraphs);

  void CopyRawPartitioningPlan(std::vector<int>& raw_plan);
  void ClearRawPartitioningPlan(){ dummy_profiles_.clear(); };

  TfLiteStatus PartitionChannels(std::vector<tflite::Subgraph*>& new_subgraphs);
  
 private:
  TfLiteStatus BuildLocalIndexToRegistrationMapping();

  // Minsung
  // override function
  TfLiteStatus BuildLocalIndexToRegistrationMapping(const ::tflite::Model* model,
                                                const OpResolver& op_resolver);

  TfLiteStatus ParseNodes(
      const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
      Subgraph* subgraph);

  // Minsung
  // override function for multiple subgraphs.
  // Parse every nodes with given op_start_index to op_end_index.
  TfLiteStatus ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Subgraph* subgraph, int op_st, int op_end);

  // Minsung
  // override function for multiple subgraphs.
  // Parse nodes with given node indicies.  
  TfLiteStatus ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Subgraph* subgraph, std::vector<int>& nodes_to_parse);

  // Minsung
  // override function for multiple subgraph
  TfLiteStatus ParseTensors(
      const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
      const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
      Subgraph* subgraph, std::vector<int> tensor_idx);


  TfLiteStatus ParseTensors(
      const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
      const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
      Subgraph* subgraph);
  TfLiteStatus ApplyDelegates(Interpreter* interpreter, int num_threads);
  TfLiteStatus ParseQuantization(const QuantizationParameters* src_quantization,
                                 TfLiteQuantization* quantization,
                                 const std::vector<int>& dims);
  TfLiteStatus ParseSparsity(const SparsityParameters* src_sparsity,
                             TfLiteSparsity** sparsity);

  const ::tflite::Model* model_;
  const OpResolver& op_resolver_;
  ErrorReporter* error_reporter_;

  std::vector<const TfLiteRegistration*> flatbuffer_op_index_to_registration_;
  std::vector<TfLiteRegistration> unresolved_custom_ops_;
  std::vector<BuiltinOperator> flatbuffer_op_index_to_registration_types_;
  const Allocation* allocation_ = nullptr;

  bool has_flex_op_ = false;
  int num_fp32_tensors_ = 0;

  // Minsung
  // name of model file
  std::string model_name_;

  // id of model
  // Every subgraphs made of this model share the same model id.
  int model_id_;
  
  int default_thread = 4;

  // flag for sub interpreter.
  bool is_sub_interpreter = false;
  ProfileData* dummy_profile_;
  std::vector<ProfileData*> dummy_profiles_;

  tflite::Interpreter* interpreter_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_INTERPRETER_BUILDER_H_

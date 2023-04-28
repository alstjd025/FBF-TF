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

// This file provides general C++ utility functions in TFLite.
// For example: Converting between `TfLiteIntArray`, `std::vector` and
// Flatbuffer vectors. These functions can't live in `context.h` since it's pure
// C.

#ifndef TENSORFLOW_LITE_UTIL_H_
#define TENSORFLOW_LITE_UTIL_H_

// Minsung
// CPU setups
#define DEFAULT_AFFINITY 0
#define DEFAULT_THREADS  1


#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {

// Memory allocation parameter used by ArenaPlanner.
// Clients (such as delegates) might look at this to ensure interop between
// TFLite memory & hardware buffers.
// NOTE: This only holds for tensors allocated on the arena.
constexpr int kDefaultTensorAlignment = 64;

// The prefix of Flex op custom code.
// This will be matched agains the `custom_code` field in `OperatorCode`
// Flatbuffer Table.
// WARNING: This is an experimental API and subject to change.
constexpr char kFlexCustomCodePrefix[] = "Flex";

// Checks whether the prefix of the custom name indicates the operation is an
// Flex operation.
bool IsFlexOp(const char* custom_name);

// Converts a `std::vector` to a `TfLiteIntArray`. The caller takes ownership
// of the returned pointer.
TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input);

// Converts an array (of the given size) to a `TfLiteIntArray`. The caller
// takes ownership of the returned pointer, and must make sure 'dims' has at
// least 'rank' elements.
TfLiteIntArray* ConvertArrayToTfLiteIntArray(const int rank, const int* dims);

// Checks whether a `TfLiteIntArray` and an int array have matching elements.
// The caller must guarantee that 'b' has at least 'b_size' elements.
bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, const int b_size,
                                 const int* b);

size_t CombineHashes(std::initializer_list<size_t> hashes);

struct TfLiteIntArrayDeleter {
  void operator()(TfLiteIntArray* a) {
    if (a) TfLiteIntArrayFree(a);
  }
};

// Helper for Building TfLiteIntArray that is wrapped in a unique_ptr,
// So that it is automatically freed when it goes out of the scope.
std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> BuildTfLiteIntArray(
    const std::vector<int>& data);

// Populates the size in bytes of a type into `bytes`. Returns kTfLiteOk for
// valid types, and kTfLiteError otherwise.
TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes);

// Creates a stub TfLiteRegistration instance with the provided
// `custom_op_name`. The op will fail if invoked, and is useful as a
// placeholder to defer op resolution.
// Note that `custom_op_name` must remain valid for the returned op's lifetime..
TfLiteRegistration CreateUnresolvedCustomOp(const char* custom_op_name);

// Checks whether the provided op is an unresolved custom op.
bool IsUnresolvedCustomOp(const TfLiteRegistration& registration);

// Returns a descriptive name with the given op TfLiteRegistration.
std::string GetOpNameByRegistration(const TfLiteRegistration& registration);


typedef struct SharedTensorsInGraphs{
  int model_id; // identifier

  // <tensor id, subgraph ids>
  std::vector<std::pair<int, std::vector<int>>> pair_tensor_graph;
} SharedTensorsInGraphs;

// Minsung
// Scheduler status
typedef enum SchedulerState{
  INIT_SCHED,           // Initial phase.
  SCHEDULING,    // Scheduling, means interpreter and jobs are ready.
  NSCHEDULABLE,         // Not schedulable.
  RESCHEDULE,     // Need reschedule
  STOP
} SchedulerStatus;

//WARNING! This struct is deprecatied.
typedef enum JobState{
  INIT_JOB,       // Initial phase. 
  READY,        // Job is ready to be scheduled.(allocation & op mapping done)
  INVOKE,      // Job is currently scheduled and invoking.
  SLEEP,       // Job is stopped by scheduler.
  DONE        // Job done and can be erased from mememory.
} JobState;

//WARNING! This struct is deprecatied.
typedef enum WorkerState{
  INIT_WORK,
  WORKING,
  BLOCKED
} WorkerState;


typedef enum ResourceType{
  CPU,
  GPU,
  CPUGPU
} ResourceType;

typedef enum RuntimeState{
  INITIALIZE,
  NEED_PROFILE,
  SUBGRAPH_CREATE,
  INVOKE_,
  BLOCKED_
} RuntimeState;

//WARNING! This struct is deprecatied.
typedef enum InvokeType{ // An invoke type of job
  CONTINOUS,          
  PROFILING
} InvokeType;

//WARNING! This struct is deprecatied.
typedef struct Job{
  int model_id = -1;      // 
  int job_id = -1;        //
  float time_slot;      //ms
  float dead_line;      //ms
  time_t start;
  time_t end;
  bool input_refreshed;
  JobState state = JobState::INIT_JOB;
  InvokeType invoke_type = InvokeType::PROFILING;
  ResourceType resource_type = ResourceType::CPU;
  std::vector<std::pair<int, int>> subgraphs;
  std::vector<int> cpu_affinity;
} Job;


typedef struct ProfileData{
  std::vector<float> latency_by_layers;
  std::vector<std::vector<int>> layer_subsets;
  std::vector<std::vector<int>> partitioning_ratios;
  std::vector<int> gpu_layers;
  bool is_valid;
} ProfileData;

typedef enum INPUT_TYPE{
  MNIST,
  IMAGENET224,
  IMAGENET300,
  USER
}INPUT_TYPE;

typedef struct tf_packet{
  short runtime_id;
  short runtime_current_state;
  short runtime_next_state;
  int cur_subgraph;
  int cur_graph_resource; // 0 for cpu, 1 for gpu
  int partitioning_plan[1000][4];
  float latency[1000];
}tf_packet;

}  // namespace tflite

#endif  // TENSORFLOW_LITE_UTIL_H_
